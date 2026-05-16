# Metal Tensor Prefill Optimization Log

Branch: `metal-tensor-prefill-next`

Date: 2026-05-14

This branch keeps the current low-drift Tensor default and uses the five-fixture
quality gate before promoting any prefill optimization.

## Drift Gate

Run:

```sh
python3 speed-bench/run_quality_drift_gate.py \
  --out-dir speed-bench/local-runs/20260514-170519-quality-drift-gate
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

Latest summary artifact:
`speed-bench/local-runs/20260514-170519-quality-drift-gate/summary.json`.

The direct equivalence test also passed:

```sh
./ds4_test --metal-mpp-equivalence
```

Result after promoting attention-output low projection to all layers while
keeping the routed-MoE Tensor window at down from layer 12 and gate/up from
layer 15:
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
  OUT_DIR=speed-bench/local-runs/20260514-160025-default-attn-out-all-compact \
  OPEN_CHART=0 \
  speed-bench/run_metal_tensor_bench.sh
```

Current Tensor default (`attn_out=all`, routed-MoE `down=12`, `up=15`,
`gate=15`) vs standard Metal:

| ctx | standard prefill t/s | tensor prefill t/s | tensor gain | standard gen t/s | tensor gen t/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 265.82 | 358.20 | 34.8% | 38.12 | 38.32 |
| 1024 | 272.46 | 373.83 | 37.2% | 37.99 | 38.07 |
| 2048 | 330.40 | 436.33 | 32.1% | 37.44 | 37.47 |
| 4096 | 341.47 | 421.93 | 23.6% | 34.35 | 34.35 |
| 8192 | 355.11 | 425.63 | 19.9% | 33.53 | 33.38 |

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
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=12` with down/up unchanged at 12/15 after the attention-output all-layer promotion | Two-repeat median vs current Tensor auto: -0.1% at 512, -0.4% at 1024, -0.7% at 2048, -2.7% at 4096, and -1.4% at 8192. Generation was within -1.1%..+0.6%. | Not run. | Reject before drift gate because moving only gate earlier is slower at every compact prefill point. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=14` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=14` with down defaulting to 12 | Two-repeat median vs down-12 Tensor auto: +2.7% at 512, +2.9% at 1024, +2.2% at 2048, +1.1% at 4096, but -0.8% at 8192. Generation was -3.2% at 8192. | Not run. | Reject before drift gate because it regresses the long-context point and generation more than the layer-15 window. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=13` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=13` with down defaulting to 12 | Two-repeat median vs current Tensor auto: -1.5% at 512, -4.0% at 1024, -2.0% at 2048, +0.9% at 4096, and +1.4% at 8192. Generation was within -2.2%..+0.2%. Artifact: `speed-bench/local-runs/20260514-172507-moe-gate-up13-down12/prefill-candidate-summary.json`. | Not run. | Reject before drift gate because it trades away short and mid-context prefill for only small long-context gains. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=14` with down defaulting to 12 and up defaulting to 15 | Two-repeat median vs current Tensor auto: -2.2% at 512, -1.7% at 1024, -0.4% at 2048, +1.0% at 4096, and +2.1% at 8192. Generation was down by 0.4%..1.9%. | Not run. | Reject before drift gate because it is a tradeoff, not a clear prefill win. |
| `DS4_METAL_MPP_MOE_UP_START_LAYER=14` with down defaulting to 12 and gate defaulting to 15 | Two-repeat median vs current Tensor auto: -3.4% at 512, -6.4% at 1024, -4.9% at 2048, -6.2% at 4096, and -5.1% at 8192. | Not run. | Reject before drift gate because it is consistently slower. |
| `DS4_METAL_MPP_MOE_TILE_N=64` | Slower than default by 3.3% to 15.6%. | Not run. | Reject before drift gate. |
| `DS4_METAL_MOE_SUM6_DISABLE=1` | Two-repeat median vs current Tensor auto: -1.6% at 512, -1.8% at 1024, -1.4% at 2048, -0.1% at 4096, and +0.6% at 8192. Generation was within -0.5%..+0.4%. | Not run. | Reject before drift gate because disabling the fused six-expert sum is slower or noise-level at every compact point. |
| `DS4_METAL_MPP_MOE_DOWN_START_LAYER=9` with gate/up unchanged at 19 | Two-repeat median vs down-12 Tensor auto: +0.3% at 512, +0.1% at 1024, -1.4% at 2048, -0.4% at 4096, and -0.5% at 8192. Generation was within -0.7%..+0.5%. | Not run. | Reject before drift gate because it is slower at most measured contexts. |
| `DS4_METAL_MPP_MOE_DOWN_START_LAYER=10` with gate/up unchanged at 19 | Two-repeat median vs 19/19/19 Tensor auto: +0.8% at 512, flat at 1024, +0.8% at 2048, +2.6% at 4096, and +2.8% at 8192. Generation was within -1.7%..+1.4%. | Five-fixture gate and `./ds4_test --metal-mpp-equivalence` passed, but `tensor_vs_standard` drift rose to worst RMS `0.314905` and worst top20 abs `0.780825`. | Not promoted because layer 12 kept useful speed with lower drift. |
| `DS4_METAL_MPP_MOE_DOWN_START_LAYER=10` with gate/up defaulting to 15 and attention-output Tensor all-layer default | Two-repeat median vs current Tensor auto: -0.1% at 512, -0.5% at 1024, -1.6% at 2048, -2.9% at 4096, and -0.8% at 8192. Generation was within -0.3%..+0.5%. | Not run. | Reject before drift gate because it is slower at every compact prefill point after the attention-output promotion. |
| `DS4_METAL_MPP_MOE_DOWN_START_LAYER=11` with gate/up unchanged at 19 | Two-repeat median vs 19/19/19 Tensor auto: +1.7% at 512, +1.7% at 1024, +3.5% at 2048, +1.7% at 4096, and +1.2% at 8192. Generation was within -1.4%..-0.3%. | Five-fixture gate passed, but `tensor_vs_standard` drift rose to worst RMS `0.314275` and worst top20 abs `0.725578`. | Not promoted because layer 12 had a better drift balance. |
| `DS4_METAL_MPP_MOE_DOWN_START_LAYER=11` with gate/up defaulting to 15 | Two-repeat median vs current Tensor auto: +0.3% at 512, -0.1% at 1024, +0.2% at 2048, +0.5% at 4096, and -2.8% at 8192. Generation was within -1.3%..+0.2%. | Not run. | Reject before drift gate because the new gate/up window removes most of the earlier speed upside and the long-context point regresses. |
| `DS4_METAL_MPP_MOE_DOWN_START_LAYER=18` with gate/up/down defaulting to 19/19/19 | Two-repeat median vs 19/19/19 Tensor auto: -2.1% at 512, -3.1% at 1024, -3.3% at 2048, -0.7% at 4096, and +1.7% at 8192. Generation was within -1.2%..+0.4%. | Not run. | Reject before drift gate because it is slower at most measured contexts. |
| `DS4_METAL_MPP_MOE_DISABLE=1` after the attention-output all-layer promotion | Two-repeat median vs current Tensor auto: -23.6% at 512, -25.0% at 1024, -22.0% at 2048, -18.0% at 4096, and -15.4% at 8192. Generation was within -1.2%..+2.4%. | Not run. | Reject before drift gate because disabling the conservative routed-MoE Tensor window removes the dominant current prefill win. |
| Local patch: route-specific routed-MoE tile env plus `DS4_METAL_MPP_MOE_DOWN_TILE_N=64` | Compact two-repeat median vs current Tensor auto: -3.3% at 512, -4.3% at 1024, -3.1% at 2048, -0.4% at 4096, and +1.7% at 8192. A one-repeat long sweep was still slightly slower from 8192..65536: -0.4%, -0.2%, -0.3%, and -0.2%. | Not run. | Reverted before drift gate because the route-specific tile knob did not produce a clear prefill win and would add another non-promotable switch. |
| `DS4_METAL_MPP_ATTN_OUT_DISABLE=1` after the attention-output all-layer promotion | Two-repeat median vs current Tensor auto: -4.6% at 512, -5.3% at 1024, -5.6% at 2048, -5.0% at 4096, and -5.1% at 8192. Generation was within -1.1%..+0.8%. | Not run. | Reject before drift gate because disabling the default all-layer attention-output Tensor route removes a clear prefill win. |
| `DS4_METAL_MPP_F16_DISABLE=1` after the attention-output all-layer promotion | Two-repeat median vs current Tensor auto: -1.1% at 512, -1.8% at 1024, -3.1% at 2048, -2.2% at 4096, and -2.5% at 8192. Generation was within -1.4%..+0.4%. | Not run. | Reject before drift gate because disabling the default F16 compressor route is slower at every compact prefill point. |
| `DS4_METAL_MPP_F16_PAIR=1` after the attention-output all-layer promotion | Two-repeat median vs current Tensor auto: -0.7% at 512, -1.1% at 1024, -0.5% at 2048, -1.8% at 4096, and -1.2% at 8192. Generation was within -1.3%..+1.1%. Artifact: `speed-bench/local-runs/20260514-171939-f16-pair-current/prefill-candidate-summary.json`. | Not run. | Reject before drift gate because it is slower at every compact prefill point. |
| `DS4_METAL_MPP_F16_WIDE=1` | Diagnostic-only wider 512/1024-column compressor Tensor route. | Existing long-code full-model equivalence check fails with wide F16 Tensor (`rms ~= 0.569`, `top20_max_abs ~= 1.48`). | Keep default-off; do not spend more prefill timing effort until the drift issue has a new mitigation. |
| `DS4_METAL_MPP_DIRECT_RHS=0` plus `DS4_METAL_MPP_F16_DIRECT_RHS=1` to isolate staged-RHS attention-output low projection | Two-repeat median vs current Tensor auto: -7.1% at 512, -4.9% at 1024, -4.5% at 2048, -3.4% at 4096, and +0.1% at 8192. Generation was within -0.6%..+0.2%. | Not run. | Reject before drift gate because it is slower at most measured contexts. Keep the direct-RHS attention-output default. |
| `DS4_METAL_MPP_ATTN_OUT_TILE_N=32` | Slower than default by 1.1% to 16.4%. | Not run. | Keep default tile 64. |
| `DS4_METAL_MPP_ATTN_OUT_FILTER=layer=31..42` | Two-repeat median vs 32..42 Tensor auto: flat at 512, then slower by 0.3% to 1.4% from 1024..8192. | Not run. | Reject before drift gate; keep attention-output at 32..42. |
| Local patch: split dense Q8_0 prefill full 32-token tiles from the non-32-token tail (`DS4_METAL_Q8_PREFILL_SPLIT_TAIL=1` prototype) | On `long_code_audit` at `ctx=3836`, two-repeat median vs current Tensor auto was +0.3% prefill and +0.6% generation. | Not run. | Reverted before drift gate because the speed change is noise-level and does not justify keeping another Q8_0 switch. |
| Local patch: dense Q8_0 cooperative Tensor direct-RHS prefill prototype scoped to `attn_q_b` | Two-repeat median vs current Tensor auto was mixed: +2.8% at 512, -1.3% at 1024, -2.2% at 2048, +2.3% at 4096, and +5.1% at 8192. Generation moved -2.5%..+0.8%. | Not run. | Reverted before drift gate because mid-context prefill and generation regressed. |
| Local patch: dense Q8_0 cooperative Tensor direct-RHS prefill prototype scoped to `attn_out`/`attn_output_b` | Two-repeat median vs current Tensor auto was +4.6% at 512, +4.4% at 1024, +6.0% at 2048, +5.2% at 4096, and +3.5% at 8192. A conservative `attn_out@layer=32..42` window was only +0.6%..+0.9% and dropped generation up to 2.2%. | All-layer `attn_out` failed the five-fixture gate: `long_memory_archive` top-1 changed and greedy differed at step 0; `tensor_vs_standard` worst RMS `0.531143` and worst top20 abs `1.17201`. | Reverted despite speed because it violates the no-new-top1/no-new-greedy rule, and the late-only safe-shape hypothesis was noise-level. |
| Local patch: paired shared-expert Q8_0 prefill matmul for `shared_gate` plus `shared_up` | Two-repeat median vs current Tensor auto: -4.8% at 512, -3.3% at 1024, -3.0% at 2048, -0.4% at 4096, and +1.4% at 8192. Generation was within -1.3%..+0.3%. Artifact: `speed-bench/local-runs/20260514-173418-shared-q8-pair-prefill/prefill-candidate-summary.json`. | Not run. | Reverted before drift gate because it slows short and mid-context prefill for only a small long-context gain. |
| `DS4_METAL_MPP_MOE_PAIR_GATE_UP=1` with gate/up/down defaulting to 19/19/19 | Two-repeat median vs 19/19/19 Tensor auto: -6.2% at 512, -3.4% at 1024, -2.7% at 2048, -2.5% at 4096, and -2.1% at 8192. Generation was within -0.2%..+1.2%. | Not run. | Reject before drift gate because the paired dispatch is consistently slower. |
| `DS4_METAL_MPP_MOE_PAIR_GATE_UP=1` after the attention-output all-layer promotion and gate/up/down defaults of 15/15/12 | Two-repeat median vs current Tensor auto: -4.0% at 512, -4.4% at 1024, -4.5% at 2048, -2.4% at 4096, and -2.5% at 8192. Generation was within -2.4%..+0.2%. | Not run. | Reject before drift gate; the paired dispatch remains slower on the wider current gate/up Tensor window. |
| Local patch: standard-Metal paired routed-MoE gate/up prefill matmul for early non-Tensor gate/up layers | Two-repeat median vs current Tensor auto: -3.8% at 512, -2.3% at 1024, -0.8% at 2048, +0.6% at 4096, and +1.3% at 8192. Generation was within -1.1%..+1.0%. Artifact: `speed-bench/local-runs/20260514-230653-experimental-moe-pair-gate-up/prefill-candidate-summary.json`. | Not run. | Reverted before drift gate. Reusing the activation tile while preserving the legacy simdgroup-MMA math did not beat separate gate/up dispatch at short and mid contexts, so it is not worth keeping as another default-off mode. |
| `DS4_METAL_MPP_MOE_FAST_LAYOUT=0` after the attention-output all-layer promotion and gate/up/down defaults of 15/15/12 | Two-repeat median vs current Tensor auto: -3.6% at 512, -3.4% at 1024, -2.3% at 2048, -1.5% at 4096, and -3.2% at 8192. Generation was within -0.5%..+0.2%. | Not run. | Reject before drift gate; the staged layout is slower than the first-PR fast layout on the current conservative window. |
| Local patch: wider non-vector FlashAttention prefill key block (`NCPSG=128` instead of 64) | One-repeat screen vs current Tensor auto: -13.1% at 512, -4.9% at 1024, -2.8% at 2048, +0.9% at 4096, and +2.7% at 8192. Generation was within -0.8%..+0.4%. Artifact: `speed-bench/local-runs/20260514-231641-flash-attn-ncpsg128/prefill-candidate-summary.json`. | Not run. | Reverted before drift gate. The larger attention key block only helps long contexts slightly and regresses the short/mid contexts that dominate the compact promotion gate. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus `DS4_METAL_EXPERIMENTAL_MOE_MATMUL_START_LAYER=18` | Two-repeat median vs current Tensor auto: +0.1% at 512, -0.1% at 1024, -0.6% at 2048, -1.8% at 4096, and -1.2% at 8192. | Not run. | Reject before drift gate because it is not faster than the current 19/19/19 default. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus `DS4_METAL_EXPERIMENTAL_MOE_MATMUL_START_LAYER=19` | Two-repeat median vs current Tensor auto: -0.9% at 512, -1.9% at 1024, -1.6% at 2048, -2.7% at 4096, and -1.8% at 8192. Generation was within -0.3%..+0.7%. | Not run. | Reject before drift gate because it is consistently slower than the current 19/19/19 default. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus `DS4_METAL_EXPERIMENTAL_MOE_MATMUL_START_LAYER=10` | Two-repeat median vs current Tensor auto: +7.5% at 512, +8.4% at 1024, +6.0% at 2048, +3.8% at 4096, +4.8% at 8192. Generation was -2.8%, -1.0%, +1.3%, +1.1%, +0.7%. | Failed the five-fixture gate: `long_memory_archive` top-1 changed and greedy differed at step 0; `tensor_vs_standard` also had one top-1 and one greedy mismatch. | Reject despite the speed because it violates the no-new-top1/no-new-greedy rule. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus `DS4_METAL_EXPERIMENTAL_MOE_MATMUL_START_LAYER=12` | Two-repeat median vs current Tensor auto: +12.2% at 512, +8.5% at 1024, +8.3% at 2048, +3.2% at 4096, +1.1% at 8192. Generation was +3.4%, -0.2%, +1.5%, -4.6%, -3.6%. | Full `./ds4_test --metal-mpp-equivalence` passed with no top-1 or greedy mismatch, but drift rose to worst RMS `0.300474` and worst top20 abs `1.00957`. | Reject before the full quality gate: long-context speed is weak and drift is much worse than the current conservative default. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus `DS4_METAL_EXPERIMENTAL_MOE_MATMUL_START_LAYER=15` | Two-repeat median vs current Tensor auto: +2.3% at 512, +2.0% at 1024, +1.5% at 2048, +2.6% at 4096, +2.0% at 8192. Generation was -2.7%, +0.0%, -1.8%, +1.1%, +1.4%. | Full `./ds4_test --metal-mpp-equivalence` passed with no top-1 or greedy mismatch, but drift rose to worst RMS `0.229322` and worst top20 abs `0.511806`. | Reject before the full quality gate: speed is marginal and drift is still worse than default. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus `DS4_METAL_EXPERIMENTAL_MOE_MATMUL_START_LAYER=17` | Two-repeat median vs current Tensor auto: +2.2% at 512, +0.5% at 1024, +0.8% at 2048, +1.2% at 4096, +0.7% at 8192. Generation was within -1.7%..+0.5%. | Full `./ds4_test --metal-mpp-equivalence` passed with no top-1 or greedy mismatch, but drift rose to worst RMS `0.190587` and worst top20 abs `0.560192`. | Reject before the full quality gate: speed is within noise and drift is worse than default. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus `DS4_METAL_MATH_SAFE=1` | Not timed. | `./ds4_test --metal-mpp-equivalence` failed: `long_memory_archive` changed top-1 and greedy at step 0; summary `top1_mismatch=1`, `greedy_fail=4`, worst RMS `0.58437`, and worst top20 abs `2.17881`. | Reject as a drift-reduction diagnostic. Strict Metal math makes the all-layer experimental route worse rather than explaining away the Tensor-vs-standard movement. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus route-specific gate start `8`, up start `15`, down start `12` | Two-repeat median vs current Tensor auto: +2.1% at 512, +2.6% at 1024, +1.5% at 2048, +1.8% at 4096, and +1.4% at 8192. Generation was within -0.6%..+0.4%. | Failed the five-fixture gate: `long_memory_archive` top-1 changed and greedy differed at step 0; `tensor_vs_standard` had one top-1 and one greedy mismatch. | Reject despite the clean timing profile because it violates the no-new-top1/no-new-greedy rule. |
| `DS4_METAL_MPP_FAST=1` plus `DS4_METAL_MPP_MOE_DOWN_START_LAYER=12` | Two-repeat median vs current Tensor auto: +13.3% at 512, +12.6% at 1024, +10.9% at 2048, +6.4% at 4096, and +6.1% at 8192. Generation had one -3.1% point at 2048 and was otherwise within -1.3%..-0.3%. Artifact: `speed-bench/local-runs/20260514-181839-mpp-fast-gate-up0-down12/`. | Failed the five-fixture gate: `tensor_vs_standard` had one greedy mismatch on `long_code_audit` (`diff@11`), with worst RMS `0.554059` and worst top20 abs `1.40659`. | Reject despite speed because it introduces a new greedy continuation change. |
| `DS4_METAL_MPP_FAST=1` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=15` plus `DS4_METAL_MPP_MOE_DOWN_START_LAYER=12` plus `DS4_METAL_MPP_MOE_DOWN_FILTER=layer=0-25,layer=27-28,layer=31-42` | Two-repeat median vs current Tensor auto: +2.0% at 512, then -1.9%, -2.1%, -2.6%, and -1.5% from 1024..8192. Generation was within -1.6%..+1.4%. Artifact: `speed-bench/local-runs/20260514-222322-mpp-fast-gate0-up15-down12-skip-down26-29-30/prefill-candidate-summary.json`. | Not run. | Reject before drift gate. Combining the fast all-layer gate route with conservative up/down windows and the known down-layer skips gives up too much compact prefill; the skipped down layers do not recover a useful speed/drift middle ground. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus route-specific gate start `0`, up start `15`, down start `12`, and `DS4_METAL_MOE_MID_F32=1` | Two-repeat median vs current Tensor auto: +4.5% at 512, +4.1% at 1024, +0.9% at 2048, -1.3% at 4096, and +0.4% at 8192. Generation was within -1.4%..-0.1%. | Not run. | Reject before drift gate because the F32 intermediate removes most of the useful route-specific prefill win and regresses the 4096-token point. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus route-specific up start `0`, gate start `15`, down start `12` | Two-repeat median vs current Tensor auto: +6.6% at 512, +6.3% at 1024, +4.5% at 2048, +3.3% at 4096, and +2.9% at 8192. Generation was within -1.4%..+0.5%. | Failed the five-fixture gate: `long_memory_archive` top-1 changed and greedy differed at step 0; `tensor_vs_standard` had one top-1 and one greedy mismatch. | Reject despite speed because it violates the no-new-top1/no-new-greedy rule. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus route-specific down start `0`, gate/up start `15` | Two-repeat median vs current Tensor auto: +4.1% at 512, +4.2% at 1024, +3.5% at 2048, +2.3% at 4096, and +2.2% at 8192. Generation was within -1.7%..+0.1%. | Failed the five-fixture gate: `long_memory_archive` top-1 changed and greedy differed at step 0; `tensor_vs_standard` had one top-1 and one greedy mismatch. | Reject despite speed because it violates the no-new-top1/no-new-greedy rule. |
| `DS4_METAL_MPP_MOE_{GATE,UP,DOWN}_START_LAYER=0` with filters adding layers 0..3 to the current default windows | Two-repeat median vs current Tensor auto: +4.4% at 512, +3.7% at 1024, +0.7% at 2048, +2.4% at 4096, and +2.0% at 8192. Generation was mostly neutral except -1.9% at 2048. Artifact: `speed-bench/local-runs/20260514-185845-mpp-gud0-3-default/`. | Failed the five-fixture gate: `tensor_vs_standard` had one greedy mismatch on `long_code_audit` (`diff@10`), with worst RMS `0.495637` and worst top20 abs `1.78119`. | Reject despite the modest speed gain because it introduces a new greedy continuation change. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=0` plus `DS4_METAL_MPP_MOE_GATE_FILTER=layer=0-3,layer=15-42`, with up/down at 15/12 | Two-repeat median vs current Tensor auto: -2.2% at 512, -2.3% at 1024, -3.5% at 2048, -1.9% at 4096, and +0.6% at 8192. Generation was within -1.2%..-0.1%. Artifact: `speed-bench/local-runs/20260514-184842-mpp-gate0-3-up15-down12/`. | Not run. | Reject before drift gate because adding only gate layers 0..3 is slower through the compact range. |
| `DS4_METAL_MPP_MOE_UP_START_LAYER=0` plus `DS4_METAL_MPP_MOE_UP_FILTER=layer=0-3,layer=15-42`, with gate/down at 15/12 | Two-repeat median vs current Tensor auto: +0.9% at 512, +0.3% at 1024, -0.4% at 2048, -2.2% at 4096, and -2.2% at 8192. Generation was within -2.1%..-0.1%. Artifact: `speed-bench/local-runs/20260514-185210-mpp-up0-3-gate15-down12/`. | Not run. | Reject before drift gate because adding only up layers 0..3 is slower at the larger compact contexts and hurts generation. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=0` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=0`, each filtered to `layer=0-3,layer=15-42`, with down defaulting to 12 | Two-repeat median vs current Tensor auto was positive: +1.7% at 512, +2.0% at 1024, +2.4% at 2048, +2.3% at 4096, and +2.6% at 8192. Generation was nearly flat, -0.4%..-0.1%. Artifact: `speed-bench/local-runs/20260515-065835-mpp-gateup0-3-down12/prefill-candidate-summary.md`. | Not run; `run_prefill_candidate_gate.py --run-drift-gate` skipped the drift gate because the repeat-level speed floor failed, with repeat prefill deltas `[-0.5%, +3.9%]` at 512 and observed min repeat prefill `-0.5%`. | Reject before drift gate. Median speed was encouraging, but the gain is not repeat-stable enough for promotion, and the speed-first guard correctly avoided a five-fixture drift run. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=0` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=0`, each filtered to `layer=0-5,layer=15-42`, with down defaulting to 12 | Two-repeat median vs current Tensor auto: +3.6% at 512, +3.0% at 1024, +1.1% at 2048, -1.2% at 4096, and +1.7% at 8192. Generation was within -1.5%..-0.1%. Artifact: `speed-bench/local-runs/20260515-070235-mpp-gateup0-5-down12/prefill-candidate-summary.md`. | Not run. | Reject before drift gate because it fails the compact speed screen at 4096 tokens and has repeat-level prefill down to -1.7%. |
| `DS4_METAL_MPP_MOE_DOWN_START_LAYER=0` plus `DS4_METAL_MPP_MOE_DOWN_FILTER=layer=0-3,layer=12-42`, with gate/up at 15/15 | Two-repeat median vs current Tensor auto: +1.5% at 512, +1.7% at 1024, -0.3% at 2048, -1.1% at 4096, and -1.3% at 8192. Generation was within -3.3%..-0.1%. Artifact: `speed-bench/local-runs/20260514-185528-mpp-down0-3-gate15-up15/`. | Not run. | Reject before drift gate because adding only down layers 0..3 regresses the larger compact contexts and generation. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=2` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=15` plus `DS4_METAL_MPP_MOE_DOWN_START_LAYER=12` | Two-repeat median vs current Tensor auto: +5.1% at 512, +4.2% at 1024, +3.9% at 2048, +2.5% at 4096, and +1.2% at 8192. Generation was within -1.5%..+0.4%. Artifact: `speed-bench/local-runs/20260514-184135-mpp-gate2-up15-down12/`. | Five-fixture gate passed, but `tensor_vs_standard` drift rose to worst RMS `0.640912` and worst top20 abs `1.11909`. | Reject because gate0/up15/down12 is faster at most points and has lower worst RMS. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=4` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=15` plus `DS4_METAL_MPP_MOE_DOWN_START_LAYER=12` | Two-repeat median vs current Tensor auto: +0.1% at 512, -1.0% at 1024, -0.5% at 2048, +1.9% at 4096, and +3.1% at 8192. Generation was within -2.0%..-0.4%. Artifact: `speed-bench/local-runs/20260514-183734-mpp-gate4-up15-down12/`. | Not run. | Reject before drift gate because it trades short/mid-context prefill and generation for only long-context gains. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=8` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=15` plus `DS4_METAL_MPP_MOE_DOWN_START_LAYER=12` | Two-repeat median vs current Tensor auto: +2.2% at 512, +2.8% at 1024, +1.9% at 2048, +1.9% at 4096, and +1.6% at 8192. Generation was within -0.8%..-0.1%. Artifact: `speed-bench/local-runs/20260514-182931-mpp-gate8-up15-down12/`. | Failed the five-fixture gate: `long_memory_archive` top-1 changed and greedy differed at step 0; `tensor_vs_standard` also had one top-1 and one greedy mismatch. | Reject because the modest speed gain is not worth the top-1 regression. |
| `DS4_METAL_MPP_FAST=1` plus `DS4_METAL_MPP_MOE_DOWN_FILTER=layer=0-25,layer=27-28,layer=32-42` | Comparator-guided follow-up after the skip-26/29/30 candidate; this also excludes `moe_down` layer 31. Two-repeat median vs current Tensor auto: +15.0% at 512, +10.9% at 1024, +8.9% at 2048, +6.0% at 4096, and +3.4% at 8192. Generation regressed by -6.1%, -3.4%, -3.5%, -3.3%, and -3.0%. Artifact: `speed-bench/local-runs/20260514-214603-mpp-fast-skip-down26-29-31/prefill-candidate-summary.md`. | Five-fixture gate failed the strict Tensor-vs-standard envelope: no top-1 or greedy mismatch, but worst RMS `0.643831` on `long_memory_archive` and worst top20 abs `1.10919` on `long_code_audit`. | Reject. Skipping layer 31 removes the remaining local `moe_down` comparator breach but does not materially reduce full-model drift, fails the generation floor at 512 tokens, and gives up too much 8192-token prefill compared with the skip-26/29/30 candidate. |
| `DS4_METAL_MPP_FAST=1` plus `DS4_METAL_MPP_MOE_DOWN_FILTER=layer=0-25,layer=27-28` | Hybrid follow-up that keeps fast all-layer gate/up Tensor but stops Tensor `moe_down` after the comparator-clean early range. Two-repeat median vs current Tensor auto: +8.5% at 512, +6.1% at 1024, +4.6% at 2048, +5.4% at 4096, and +5.9% at 8192. Generation was within -1.0%..+0.6%. Artifact: `speed-bench/local-runs/20260515-023038-mpp-fast-gate-up0-down-clean-early/prefill-candidate-summary.md`. | Five-fixture gate failed the strict Tensor-vs-standard envelope: no top-1 or greedy mismatch, but worst RMS `0.643635` on `long_memory_archive` and worst top20 abs `1.11349` on `long_code_audit`. | Reject. Removing late `moe_down` Tensor does not fix the route-wide drift, and it is slower than the skip-26/29/30 default-off candidate. |

## Promoted Candidates

| Candidate | Speed result | Drift result | Decision |
| --- | --- | --- | --- |
| `DS4_METAL_MPP_ATTN_OUT_FILTER=all` | Two-repeat median vs current Tensor auto: +3.1% at 512, +3.3% at 1024, +3.6% at 2048, +2.2% at 4096, and +2.1% at 8192. Generation was within -1.1%..+0.3%. | Five-fixture gate passed and `./ds4_test --metal-mpp-equivalence` passed. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1`, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.239946`, worst top20 abs `0.55422`, matching the current default envelope. | Promoted: attention-output low projection now defaults to all layers; `late_safe` remains available for the old 32..42 window. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=19` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=19` plus `DS4_METAL_MPP_MOE_DOWN_START_LAYER=21` | Two-repeat median vs current Tensor auto: +0.6% at 512, +0.8% at 1024, +2.3% at 2048, +2.0% at 4096, +1.6% at 8192. Generation was within -1.4%..+0.5%. | Five-fixture gate passed, first as env candidate and again as the env-free default after promotion. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1` matching standard-vs-quality, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.176030`, worst top20 abs `0.360397`. | Promoted, then superseded by the lower-drift 19/19/20 window and the faster 19/19/19 window. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=19` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=19` plus `DS4_METAL_MPP_MOE_DOWN_START_LAYER=20` | Two-repeat median vs 19/19/21 Tensor auto: +0.3% at 512, +1.2% at 1024, +0.9% at 2048, +0.4% at 4096, +0.2% at 8192. Generation was within -0.9%..+1.0%. | Five-fixture gate passed, first as env candidate and again as the env-free default after promotion. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1` matching standard-vs-quality, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.066747`, worst top20 abs `0.191437`. | Promoted, then superseded by the slightly faster 19/19/19 window. |
| `DS4_METAL_MPP_MOE_DOWN_START_LAYER=19` with gate/up unchanged at 19 | Two-repeat median vs 19/19/20 Tensor auto: +0.9% at 512, +1.2% at 1024, +1.1% at 2048, +0.4% at 4096, +0.9% at 8192. Generation was within -1.0%..+1.4%. | Five-fixture env-candidate gate passed. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1` matching standard-vs-quality, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.136143`, worst top20 abs `0.315292`. | Promoted as the next routed-MoE default window: gate/up/down from layer 19. |
| `DS4_METAL_MPP_MOE_DOWN_START_LAYER=12` with gate/up unchanged at 19 | Two-repeat median vs 19/19/19 Tensor auto: +2.1% at 512, +0.8% at 1024, +2.0% at 2048, +1.1% at 4096, and +1.5% at 8192. Env-free compact timing after promotion showed Tensor prefill +26.7%, +28.8%, +21.9%, +18.7%, and +15.7% vs standard Metal from 512..8192. | Five-fixture env-candidate gate and env-free default gate passed. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1` matching standard-vs-quality, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.229474`, worst top20 abs `0.601166`. `./ds4_test --metal-mpp-equivalence` also passed with the same worst RMS/top20 abs. | Promoted, then superseded by the layer-15 gate/up window. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=15` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=15` with down defaulting to 12 | Two-repeat median vs down-12 Tensor auto: +2.2% at 512, +1.5% at 1024, +0.3% at 2048, +0.2% at 4096, and +0.6% at 8192. Env-free compact timing after promotion shows Tensor prefill +32.3%, +31.7%, +24.7%, +19.8%, and +17.0% vs standard Metal from 512..8192. | Five-fixture env-candidate gate and env-free default gate passed. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1` matching standard-vs-quality, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.239946`, worst top20 abs `0.55422`. `./ds4_test --metal-mpp-equivalence` also passed with the same worst RMS/top20 abs. | Promoted as the current routed-MoE default window: down from layer 12, gate/up from layer 15. |

## Default-Off Candidates

| Candidate | Speed result | Drift result | Decision |
| --- | --- | --- | --- |
| `DS4_METAL_MPP_FAST=1` | Post-attention-output-promotion two-repeat median vs current Tensor auto: +18.1% at 512, +18.3% at 1024, +12.3% at 2048, +7.4% at 4096, and +7.1% at 8192. Generation was neutral, within -0.1%..+0.7%. | Five-fixture gate passed and `./ds4_test --metal-mpp-equivalence` passed. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1`, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.669241`, worst top20 abs `1.30664`. | Keep default-off as the strongest speed/eval candidate. It widens routed-MoE Tensor to layer 0, but the Tensor-vs-standard drift is much larger than the conservative default. |
| `DS4_METAL_MPP_FAST=1` plus `DS4_METAL_MPP_MOE_DOWN_FILTER=layer=0-25,layer=27-42` | Two-repeat median vs current Tensor auto: +15.8% at 512, +14.6% at 1024, +9.4% at 2048, +9.0% at 4096, and +9.6% at 8192. Generation was within -0.8%..+0.0%. Artifact: `speed-bench/local-runs/20260514-180751-mpp-fast-skip-down26/`. | Five-fixture gate passed. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1`, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.645033`, worst top20 abs `1.28496`. | Keep default-off. Skipping the local comparator outlier layer 26 trims the fast-route drift slightly but remains far above the conservative default drift envelope. |
| `DS4_METAL_MPP_FAST=1` plus `DS4_METAL_MPP_MOE_DOWN_FILTER=layer=0-25,layer=27-28,layer=31-42` | Two-repeat median vs current Tensor auto: +19.3% at 512, +19.5% at 1024, +7.8% at 2048, +6.1% at 4096, and +6.0% at 8192. Generation was mixed but acceptable for a prefill-first candidate: +1.7%, +0.5%, -3.5%, -2.5%, and +1.8%. Artifact: `speed-bench/local-runs/20260514-212340-mpp-fast-skip-down26-29-30/prefill-candidate-summary.json`. | Five-fixture gate passed. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1`, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.643810`, worst top20 abs `1.13945`. `./ds4_test --metal-mpp-equivalence` also passed with the same Tensor summary. | Keep default-off as the best current eval candidate. Comparator-guided exclusions remove the large `moe_down` local outliers at layers 26, 29, and 30, reducing top20 Tensor-vs-standard drift versus the layer-26-only skip while keeping a larger compact prefill win. |
| `DS4_METAL_MPP_FAST=1` plus `DS4_METAL_MPP_MOE_DOWN_FILTER=layer=0-25,layer=27-28,layer=31-42` plus `DS4_METAL_MOE_MID_F32=1` | Two-repeat median vs current Tensor auto: +12.0% at 512, +11.5% at 1024, +6.7% at 2048, +4.9% at 4096, and +6.1% at 8192. Generation was flatter than the F16-mid skip candidate: -0.2%, -1.4%, -1.1%, -0.8%, and -0.7%. Artifact: `speed-bench/local-runs/20260514-222853-mpp-fast-skip-down26-29-30-mid-f32/prefill-candidate-summary.json`. | Five-fixture gate passed. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1`, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.643810`, worst top20 abs `1.13945`. `./ds4_test --metal-mpp-equivalence` also passed with the same Tensor summary. | Keep default-off as the best balanced eval candidate when generation steadiness matters. It gives up some short-context prefill versus the F16-mid skip candidate but keeps long-context prefill similar and avoids the larger generation timing swings. |
| `DS4_METAL_MPP_FAST=1` plus `DS4_METAL_MPP_MOE_DOWN_FILTER=layer=0-23,layer=25,layer=27-42` | Two-repeat median vs current Tensor auto: +18.4% at 512, +18.0% at 1024, +12.4% at 2048, +10.1% at 4096, and +8.1% at 8192. Generation was within -1.5%..-0.1%. Artifact: `speed-bench/local-runs/20260514-181319-mpp-fast-skip-down24-26/`. | Five-fixture gate passed. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1`, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.645334`, worst top20 abs `1.44783`. | Keep default-off, but prefer the layer-26-only skip if using this diagnostic because it has lower top20 drift. |
| `DS4_METAL_MPP_FAST=1` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=15` plus `DS4_METAL_MPP_MOE_DOWN_START_LAYER=12` | Two-repeat median vs current Tensor auto: +6.1% at 512, +5.0% at 1024, +4.0% at 2048, +2.7% at 4096, and +2.8% at 8192. Generation was within -1.0%..+0.2%. Artifact: `speed-bench/local-runs/20260514-182359-mpp-fast-gate0-up15-down12/`. | Five-fixture gate passed. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1`, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.529461`, worst top20 abs `1.05153`. | Keep default-off. It is the cleanest new route-split gate result, but the Tensor-vs-standard drift is still materially larger than the current default for only a modest speed gain. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` | Two-repeat median vs current Tensor auto: +15.9% at 512, +19.7% at 1024, +12.5% at 2048, +6.8% at 4096, +11.7% at 8192. Generation was -4.9%, -1.5%, -3.5%, -0.9%, -1.7%. | Five-fixture gate passed. `tensor_vs_quality` stayed inside the current standard-vs-quality envelope with top1 mismatches `0`, greedy mismatches `1`, worst RMS `0.618172`, and worst top20 abs `2.24006`. `tensor_vs_standard` had no top1 or greedy mismatch, but drift increased to worst RMS `0.669241` and worst top20 abs `1.30664`. | Keep default-off until an eval confirms the larger Tensor-vs-standard logit movement is acceptable. This is the best prefill candidate so far, but not yet promoted over the lower-drift conservative default. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus `DS4_METAL_MOE_MID_F32=1` | Two-repeat median vs current Tensor auto: +10.8% at 512, +11.8% at 1024, +6.0% at 2048, +4.0% at 4096, and +6.0% at 8192. Generation was neutral, within -0.5%..+0.3%. | Five-fixture gate passed and `./ds4_test --metal-mpp-equivalence` passed. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1`, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.669241`, worst top20 abs `1.30664`. | Keep default-off. The F32 MoE intermediate improves generation timing versus the all-layer experimental route, but it does not reduce the larger Tensor-vs-standard drift and gives up part of the prefill win. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus route-specific gate start `0`, up start `15`, down start `12` | Two-repeat median vs current Tensor auto: +2.0% at 512, +4.6% at 1024, +6.1% at 2048, +7.3% at 4096, and +4.6% at 8192. Generation was near flat through 4096 and -4.4% at 8192. | Five-fixture gate passed. `tensor_vs_quality` stayed inside the current standard-vs-quality envelope with top1 mismatches `0`, greedy mismatches `1`, worst RMS `0.618172`, and worst top20 abs `2.24006`. `tensor_vs_standard` had no top1 or greedy mismatch, but drift rose to worst RMS `0.529461` and worst top20 abs `1.05153`. | Keep default-off. It is the best route-specific speed candidate that still passes the gate, but it is not promoted because Tensor-vs-standard drift is materially larger than the current conservative default and the 8192 generation point regressed in timing. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus route-specific gate start `0`, up start `15`, down start `12`, after the attention-output all-layer promotion | Two-repeat median vs current Tensor auto: +5.6% at 512, +5.3% at 1024, +4.3% at 2048, +1.6% at 4096, and +0.3% at 8192. Generation was within -0.6%..+0.8%. | Not rerun after the attention-output promotion because the same route already passed the five-fixture gate before promotion and the speed profile is not strong enough to promote. | Keep default-off. The current default absorbed most of the long-context prefill benefit, leaving this as a short-context diagnostic rather than a production default. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus `DS4_METAL_MPP_MOE_FAST_LAYOUT=0` | Two-repeat median vs current Tensor auto: +8.4% at 512, +12.3% at 1024, +0.4% at 2048, +1.2% at 4096, and +4.3% at 8192. Generation was -4.2% at 1024, -3.2% at 2048, -4.4% at 4096, and near flat at 512/8192. | Five-fixture gate passed, but `tensor_vs_standard` was unchanged from the faster experimental layout: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.669241`, and worst top20 abs `1.30664`. | Reject as the preferred experimental layout because it gives up speed without reducing the larger Tensor-vs-standard movement. |

## Profile Signal

`speed-bench/run_prefill_candidate_gate.py` now has named `--preset` values for
the measured default-off profiles, including `mpp-fast`,
`mpp-fast-skip-down26-29-30`,
`mpp-fast-skip-down26-29-30-mid-f32`, and
`experimental-moe-matmul`. Explicit `--set-env` values still override the preset.
This keeps future speed/drift reruns tied to the same five-fixture gate while
removing long env strings from the critical path.

The preset table is shared through `speed-bench/metal_tensor_presets.py`, and
`speed-bench/run_quality_drift_gate.py` now accepts the same `--preset` option
for standalone five-fixture logprob checks. A preset drift run stores artifacts
under `speed-bench/local-runs/<datetime>-<preset>-quality-drift-gate/` by
default. This makes the drift-only rerun for the current best candidate:

```sh
python3 speed-bench/run_quality_drift_gate.py \
  --preset mpp-fast-skip-down26-29-30 \
  --max-tensor-standard-rms 0.30 \
  --max-tensor-standard-top20-abs 0.60
```

`speed-bench/summarize_mpp_compare.py` now parses `DS4_METAL_MPP_COMPARE_*`
logs into Markdown and JSON. The existing best-candidate comparator log was
regenerated as:

- `speed-bench/local-runs/20260515-014911-mpp-compare-summary/mpp-compare-summary.md`
- `speed-bench/local-runs/20260515-014911-mpp-compare-summary/mpp-compare-summary.json`

The summary preserves the key local attribution: the first comparator target
breach in that run is `moe_down` at layer 31 with max abs `0.00341797` and RMS
`2.5071e-06`; the next-largest local deltas are well below the comparator max
abs target. This supports keeping the skip-26/29/30 candidate default-off rather
than promoting or widening it without an eval.

A follow-up `--all-cases --route moe_down` comparator probe on the same
skip-26/29/30 preset confirmed that layer 31 is the only remaining local
`moe_down` target breach in the five fixtures, and it appears only in the two
long prompts:

- `speed-bench/local-runs/20260515-020415-mpp-fast-skip-down26-29-30-mpp-compare-probe/mpp-compare-summary.md`

Excluding layer 31 as well (`layer=0-25,layer=27-28,layer=32-42`) was then
rerun through the five-fixture drift gate. It still failed the strict
Tensor-vs-standard envelope with worst RMS `0.643831` and worst top20 abs
`1.10919`, while the speed scorecard failed the generation floor at 512 tokens.
That means the remaining full-model movement is not fixed by skipping the one
remaining local down-layer breach.

`speed-bench/run_mpp_compare_probe.py` now wraps this comparator workflow:

```sh
python3 speed-bench/run_mpp_compare_probe.py \
  --preset mpp-fast-skip-down26-29-30 \
  --case long_memory_archive \
  --route moe_down
```

It uses the same preset table, writes raw logs and `mpp-compare-summary.md/json`
under ignored `speed-bench/local-runs/`, and supports `--all-cases` for the
same five fixtures used by `run_quality_drift_gate.py`. `--route` is repeatable
and accepts comma or pipe separated lists, but each route is run separately
because the underlying comparator accepts one route at a time. This should be
used only for local attribution before the logprob gate, not as a promotion
signal.

`speed-bench/run_prefill_candidate_gate.py --run-drift-gate` now enforces the
speed-first workflow: it evaluates the compact prefill/generation speed screen
before launching the five-fixture drift gate, and records a skip reason instead
of spending a drift run on candidates that already fail the speed floor. This
keeps local optimization sweeps aligned with the promotion rule: speed screen
first, drift gate only for speed-positive candidates.

Best default-off skip-26/29/30 profile:

```sh
env DS4_METAL_MPP_FAST=1 \
    DS4_METAL_MPP_MOE_DOWN_FILTER=layer=0-25,layer=27-28,layer=31-42 \
    DS4_METAL_GRAPH_TOKEN_PROFILE=1 \
    DS4_METAL_LAYER_STAGE_PROFILE=1 \
    DS4_METAL_MOE_STAGE_PROFILE=1 \
    DS4_METAL_ATTN_OUT_STAGE_PROFILE=1 \
    DS4_METAL_Q8_PREFILL_PROFILE=1 \
    DS4_METAL_Q8_PREFILL_PROFILE_FILTER=attn_ \
    ./ds4 --metal -mt auto \
      --prompt-file tests/test-vectors/prompts/long_code_audit.txt \
      -c 8192 -n 1 --system "" --nothink --temp 0
```

Output:

`speed-bench/local-runs/20260514-214926-mpp-fast-skip26-29-30-profile/long_code_audit_profile.stderr`

This diagnostic run reported `prefill: 397.46 t/s`. With stage-level flushes
enabled, use these numbers for attribution rather than throughput comparison.

Important medians at `tokens=3844`, excluding layer 0 first-use overhead:

- Dense attention Q8_0: `attn_q_a=2.947 ms`, `attn_kv=1.621 ms`,
  `attn_q_b=21.102 ms`, and `attn_out=21.683 ms`.
- Routed-MoE Tensor layers (`mpp=1/1/1`, 39 layers): gate `16.386 ms`, up
  `16.558 ms`, down `15.795 ms`.
- Skipped-down layers (`mpp=1/1/0`, layers 26/29/30): gate `16.623 ms`, up
  `16.480 ms`, legacy down `37.776 ms`.
- Layer-stage medians: attention `43.248 ms`, attention output projection
  `43.636 ms`, routed MoE `51.724 ms`, shared gate/up `11.070 ms`, and shared
  down `7.975 ms`.

This makes dense attention `attn_q_b` and `attn_output_b` the next meaningful
kernel target after the route-window work. Further down-layer exclusions reduce
local comparator outliers but start to give up too much generation and
long-context prefill speed.

## Long-Context Candidate Validation

The current strongest passing default-off speed candidate was also measured in
a one-repeat full sweep with 128 generated tokens:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --candidate-label mpp-fast-skip-down26-29-30-long128 \
  --ctx-max 65536 \
  --gen-tokens 128 \
  --repeat 1 \
  --set-env DS4_METAL_MPP_FAST=1 \
  --set-env DS4_METAL_MPP_MOE_DOWN_FILTER=layer=0-25,layer=27-28,layer=31-42
```

Artifact:
`speed-bench/local-runs/20260514-212917-mpp-fast-skip-down26-29-30-long128/prefill-candidate-summary.json`.

| ctx | candidate prefill vs current Tensor | candidate gen vs current Tensor |
| ---: | ---: | ---: |
| 512 | +15.1% | -0.1% |
| 1024 | +15.3% | -0.5% |
| 2048 | +11.4% | -0.2% |
| 4096 | +8.3% | +1.0% |
| 8192 | +8.7% | -0.4% |
| 16384 | +7.2% | -0.2% |
| 32768 | +6.1% | -0.4% |
| 65536 | +5.8% | -0.3% |

Decision remains default-off: the full sweep confirms a real prefill win across
the long range, and the five-fixture gate is clean, but Tensor-vs-standard drift
is still materially larger than the conservative default. This is the best eval
candidate if we decide to test whether the larger Tensor-vs-standard movement
is acceptable in task-level quality.

The balanced F32-mid variant was measured in the same long sweep shape:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --candidate-label mpp-fast-skip-down26-29-30-mid-f32-long128 \
  --ctx-max 65536 \
  --gen-tokens 128 \
  --repeat 1 \
  --set-env DS4_METAL_MPP_FAST=1 \
  --set-env DS4_METAL_MPP_MOE_DOWN_FILTER=layer=0-25,layer=27-28,layer=31-42 \
  --set-env DS4_METAL_MOE_MID_F32=1
```

Artifact:
`speed-bench/local-runs/20260514-223632-mpp-fast-skip-down26-29-30-mid-f32-long128/prefill-candidate-summary.json`.

| ctx | candidate prefill vs current Tensor | candidate gen vs current Tensor |
| ---: | ---: | ---: |
| 512 | +15.9% | -1.1% |
| 1024 | +11.1% | -1.5% |
| 2048 | +6.7% | -1.5% |
| 4096 | +7.2% | -0.8% |
| 8192 | +5.1% | -0.9% |
| 16384 | +5.0% | -0.3% |
| 32768 | +2.6% | -1.5% |
| 65536 | +2.4% | -2.7% |

Decision remains default-off and secondary to the faster F16-mid skip candidate
for pure prefill. The balanced variant still gives a real prefill win across
the full range and passed the five-fixture gate plus
`./ds4_test --metal-mpp-equivalence`, but gives up the strongest long-context
prefill gains and has a -2.7% generation point at 65536. Use it only when the
flatter compact generation profile is more important than maximum prefill.

The earlier layer-26-only skip candidate was measured in the same shape:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --candidate-label mpp-fast-skip-down26-long128 \
  --ctx-max 65536 \
  --gen-tokens 128 \
  --repeat 1 \
  --set-env DS4_METAL_MPP_FAST=1 \
  --set-env DS4_METAL_MPP_MOE_DOWN_FILTER=layer=0-25,layer=27-42
```

Artifact:
`speed-bench/local-runs/20260514-190526-mpp-fast-skip-down26-long128/prefill-candidate-summary.json`.

| ctx | candidate prefill vs current Tensor | candidate gen vs current Tensor |
| ---: | ---: | ---: |
| 512 | +18.3% | +0.2% |
| 1024 | +12.4% | -1.1% |
| 2048 | +6.2% | -2.0% |
| 4096 | +6.3% | -0.6% |
| 8192 | +5.6% | -0.7% |
| 16384 | +5.7% | -0.1% |
| 32768 | +4.7% | -0.4% |
| 65536 | +6.9% | -0.0% |

Decision remains default-off: the full sweep confirms a real prefill win across
the long range, but the five-fixture gate still shows much larger
Tensor-vs-standard drift than the conservative default. The newer
skip-26/29/30 candidate above keeps a stronger long-context prefill profile at
most measured contexts and lower top-20 Tensor-vs-standard drift, so prefer that
one for any task-level eval.

The smaller `gate0/up15/down12` passing candidate was also measured in the same
long sweep shape:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --candidate-label mpp-fast-gate0-up15-down12-long128 \
  --ctx-max 65536 \
  --gen-tokens 128 \
  --repeat 1 \
  --set-env DS4_METAL_MPP_FAST=1 \
  --set-env DS4_METAL_MPP_MOE_UP_START_LAYER=15 \
  --set-env DS4_METAL_MPP_MOE_DOWN_START_LAYER=12
```

Artifact:
`speed-bench/local-runs/20260514-191816-mpp-fast-gate0-up15-down12-long128/prefill-candidate-summary.json`.

| ctx | candidate prefill vs current Tensor | candidate gen vs current Tensor |
| ---: | ---: | ---: |
| 512 | +4.4% | -0.8% |
| 1024 | -0.3% | -4.2% |
| 2048 | +1.1% | -1.0% |
| 4096 | +1.3% | -0.1% |
| 8192 | +1.6% | -1.4% |
| 16384 | +0.6% | -0.9% |
| 32768 | +0.3% | -0.4% |
| 65536 | -3.9% | -8.0% |

Decision: reject for long-context promotion. The compact gate passed, but the
full sweep shows it is noise-level for prefill and regresses generation at the
largest context.

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

Output:

`speed-bench/local-runs/20260514-161802-current-default-attn-all-profile/long_code_audit_profile.log`

Current default diagnostic result: `prefill: 414.91 t/s`. This run enables
stage-level flushes for attribution; use the compact timing chart above as the
primary speed comparison.

Important stage timings at `tokens=3844`:

- Layers 0..11 use legacy routed-MoE projections (`mpp=0/0/0`): median gate
  `33.420 ms`, up `34.368 ms`, down `33.380 ms`.
- Layers 12..14 use Tensor down only (`mpp=0/0/1`): median gate `33.334 ms`,
  up `33.355 ms`, down `13.748 ms`.
- Layers 15..42 use Tensor gate/up/down (`mpp=1/1/1`): median gate
  `14.343 ms`, up `14.372 ms`, down `13.822 ms`.
- Dense attention Q8_0 medians are `attn_q_a=2.523 ms`,
  `attn_kv=1.415 ms`, `attn_q_b=18.507 ms`, and `attn_out=18.821 ms`.
- The attention output projection stage remains about `38.017 ms/layer`;
  with all-layer attention-output Tensor enabled, the low projection is
  `19.153 ms` and the output projection is `18.906 ms`.

Shared-expert dense Q8_0 profile:

`speed-bench/local-runs/20260514-173017-shared-q8-profile/long_code_audit.stderr`

- On `long_code_audit`, `tok=3844`, median `shared_gate` was `4.701 ms`,
  `shared_up` was `4.691 ms`, and `shared_down` was `4.702 ms`.
- The median combined shared-expert dense Q8_0 time was `14.284 ms/layer`.
- A paired `shared_gate`/`shared_up` prefill prototype was tested and reverted;
  it was slower through 4096 tokens and only slightly faster at 8192.

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
attention targets remain `attn_q_b in=1024 out=32768` and the second attention
output projection `attn_output_b`.

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

A wider comparator run on `long_memory_archive` with
`DS4_METAL_MPP_COMPARE_MAX=200` did find the first local breach in `moe_down`
layer 26: max abs `0.00109863`, RMS `1.12718e-06`
(`speed-bench/local-runs/20260514-174248-experimental-moe-compare/`). Earlier
gate/up rows were around `1e-5` to `1e-4`, so the next routed-MoE experiment
should keep the down route scoped and treat wider down windows as drift risk.

The same long fixture with the passing `gate0/up15/down12` split and
`DS4_METAL_MPP_COMPARE_ROUTE=moe_gate` did not show a single bad gate layer:
all gate local max abs values stayed around `1e-5` to `6e-5`
(`speed-bench/local-runs/20260514-184759-gate0-route-compare/`). This points
to accumulated model movement from widening the gate route, not one obvious
gate-layer exclusion candidate.

Comparator follow-up on the current best skip-26/29/30 candidate:

```sh
env DS4_METAL_MPP_FAST=1 \
    DS4_METAL_MPP_MOE_DOWN_FILTER=layer=0-25,layer=27-28,layer=31-42 \
    DS4_METAL_MPP_COMPARE_MAX=100 \
    DS4_METAL_MPP_COMPARE_ROUTE=moe_gate|moe_up \
    ./ds4 --metal -mt auto \
      --prompt-file tests/test-vectors/prompts/long_memory_archive.txt \
      -c 16384 -n 1 --system "" --nothink --temp 0
```

Artifacts:

- `speed-bench/local-runs/20260514-225400-mpp-fast-skip26-29-30-gate-comparator-max100/`
- `speed-bench/local-runs/20260514-225400-mpp-fast-skip26-29-30-up-comparator-max100/`

Neither `moe_gate` nor `moe_up` reported a local comparator breach over the
available comparisons. This makes another gate/up layer-exclusion pass
unlikely to improve the speed/drift tradeoff; the known actionable local
outliers were the `moe_down` layers already excluded by the skip-26/29/30
candidate.

`DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` with gate/up from layer 0 and down from
layer 12 was benchmarked as
`speed-bench/local-runs/20260514-174353-experimental-gate-up0-down12/`. It was
not a clean speed candidate versus the current Tensor default: prefill changed
by `-6.0%`, `-6.7%`, `-5.6%`, `-5.3%`, and `+2.1%` for contexts 512..8192,
while generation changed by `-11.0%`, `-8.2%`, `-6.3%`, `-4.4%`, and `-1.1%`.
This was rejected before running the drift gate.

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

## Stage Profile Summarizer

Added `speed-bench/summarize_stage_profile.py` to convert Metal layer, routed
MoE, attention-output, and Q8 prefill profile logs into a ranked Markdown/JSON
summary. It is a local analysis helper only; summaries should be written under
`speed-bench/local-runs/`.

Current snapshot:

- `speed-bench/local-runs/20260514-231404-stage-profile-summary/stage-profile-summary.md`
- `speed-bench/local-runs/20260514-231404-stage-profile-summary/stage-profile-summary.json`

The current conservative profile on `long_code_audit` ranks parsed stages as
`ffn.routed_moe=2790.479 ms`, `attn.attention=1760.972 ms`,
`attn.output_proj=1638.645 ms`, and `attn.q_path=1165.267 ms`.
Nested profile lines overlap, so these are ranking signals rather than
exclusive wall-time shares. After the routed-MoE route-window and dense-Q8
prototype boundaries below, the remaining non-repeated performance target is
the compressed/prefill attention kernel itself. The first simple shape test,
widening non-vector FlashAttention from 64 to 128 key rows per group, was
rejected before drift gating because it regressed compact short and mid
contexts.

## FlashAttention Stage Profiler

Artifact root:

- `speed-bench/local-runs/20260514-232644-flash-attn-stage-profile/`

Patch added a default-off `DS4_METAL_FLASH_ATTN_STAGE_PROFILE=1` profiler for
raw and static-mixed prefill FlashAttention helpers. The profiler splits GPU
batches at stage boundaries and updates the wrapper-owned command buffer, so it
does not affect normal execution when the env var is unset.

Smoke command:

```sh
DS4_METAL_FLASH_ATTN_STAGE_PROFILE=1 ./ds4-bench \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 512 \
  --ctx-max 512 \
  --step-mul 2 \
  --gen-tokens 1 \
  -mt auto \
  --csv speed-bench/local-runs/20260514-232644-flash-attn-stage-profile/smoke.csv
```

Summarized profile:

| Stage | total ms | events | avg ms |
| --- | ---: | ---: | ---: |
| `flash_attn.static_mixed_nonvec.attention` | 78.117 | 41 | 1.905 |
| `flash_attn.static_mixed_nonvec.copy_raw` | 8.332 | 41 | 0.203 |
| `flash_attn.static_mixed_nonvec.copy_comp` | 7.821 | 41 | 0.191 |
| `flash_attn.static_mixed_nonvec.block_map` | 7.209 | 41 | 0.176 |
| `flash_attn.raw_nonvec.attention` | 4.516 | 2 | 2.258 |
| `flash_attn.static_mixed_nonvec.mask_fill` | 4.489 | 41 | 0.109 |
| `flash_attn.static_mixed_nonvec.pad` | 4.124 | 20 | 0.206 |

Shape split:

| FlashAttention shape | total ms | events | avg ms |
| --- | ---: | ---: | ---: |
| `static_mixed_nonvec tokens=512 comp=128 keys=640 heads=64 dim=512 window=128 ratio=4` | 56.452 | 105 | 0.538 |
| `static_mixed_nonvec tokens=512 comp=4 keys=516 heads=64 dim=512 window=128 ratio=128` | 53.640 | 120 | 0.447 |
| `raw_nonvec tokens=512 comp=0 keys=512 heads=64 dim=512 window=128 ratio=0` | 5.825 | 8 | 0.728 |

Conclusion: after routed-MoE and attention-output work, the prefill attention
kernel itself is the next high-signal target. Copy, mask, block-map, and pad
costs are visible but secondary in this smoke; a real optimization attempt
should focus on the non-vector static-mixed attention kernel and keep the
five-fixture drift gate as the promotion check.

## Rejected FlashAttention Tile Variants

Artifact roots:

- `speed-bench/local-runs/20260514-233823-flash-attn-c32-real/`
- `speed-bench/local-runs/20260514-234143-flash-attn-q16-real/`

Two real non-vector prefill FlashAttention specializations were tested after
the stage profiler pointed at `static_mixed_nonvec.attention`:

- `C=32`, `Q=8`, `NSG=4`;
- `Q=16`, `C=64`, `NSG=8`.

Both used matching attention, pad, and block-map tile sizes in the tested local
patch. Earlier host-only screens for `C=32` and `Q=16` were discarded because
the exported attention kernel is template-specialized for `Q=8,C=64`; changing
only host pad/block constants is not a valid candidate.

Compact two-repeat medians versus current Tensor auto:

| Candidate | 512 | 1024 | 2048 | 4096 | 8192 | Generation impact |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| real `C=32` | -9.5% | -5.0% | -5.4% | -3.1% | +0.5% | -1.5% to flat |
| real `Q=16` | -8.7% | +0.8% | +0.3% | -0.2% | -0.3% | -1.7% to -0.1% |

Decision: revert/no production knob and no drift gate. The corrected
specializations did not meet the speed bar, so the next attention attempt needs
a real kernel design change rather than changing only the query/key tile
geometry.

## Routed-MoE Prototype Boundary

Current routed-MoE prefill already has these measured Metal 4 variants:

- default conservative Tensor window: down from layer 12, gate/up from layer 15;
- `DS4_METAL_MPP_FAST=1`: all-layer routed-MoE Tensor;
- route-specific windows and filters for gate/up/down;
- `DS4_METAL_MPP_MOE_TILE_N=64`;
- `DS4_METAL_MPP_MOE_FAST_LAYOUT=0`;
- `DS4_METAL_MPP_MOE_PAIR_GATE_UP=1`;
- a local standard-Metal paired gate/up kernel that kept the legacy simdgroup
  reduction shape but reused the activation tile;
- `DS4_METAL_MOE_MID_F32=1`.

The useful default-off frontier is now the skip-26/29/30 family:

- fastest prefill: `DS4_METAL_MPP_FAST=1` plus
  `DS4_METAL_MPP_MOE_DOWN_FILTER=layer=0-25,layer=27-28,layer=31-42`;
- balanced generation: same env plus `DS4_METAL_MOE_MID_F32=1`.

Both pass the five-fixture gate and `./ds4_test --metal-mpp-equivalence`, but
they remain default-off because Tensor-vs-standard drift is materially larger
than the conservative default. Additional gate/up exclusion scans on the
fastest skip candidate did not find local comparator breaches, and excluding
more down layers, such as layer 31, gave up too much generation and long-context
prefill speed. A later hybrid that disabled all late `moe_down` Tensor while
keeping fast gate/up Tensor still failed the strict Tensor-vs-standard envelope,
which reinforces that the remaining movement is route-wide rather than a single
late down-layer issue.

Conclusion: env-only routed-MoE tuning is exhausted for this branch. The next
routed-MoE optimization should be a real kernel design change, not another
route-window combination. A useful design target would preserve the current
fast-layout speed while reducing accumulated full-model movement from the
all-layer gate/up/down window, with the route comparator and five-fixture gate
as hard promotion checks.

## Early Routed-MoE Kernel Contract

Inspection target:

- `metal/moe.metal`: `kernel_mul_mm_id`, `kernel_mul_mm_id_mpp_fast_layout`,
  and `kernel_mul_mm_id_pair_mpp`.
- `ds4_metal.m`: `ds4_gpu_routed_mm_pipeline`,
  `ds4_gpu_encode_mul_mm_id_map`, and the routed batch MoE dispatch around
  `ds4_gpu_encode_mul_mm_id_mapped_tile`.

Current dispatch already does the right high-level batching:

- one expert-major route map is built per layer and reused for gate, up, and
  down;
- gate and up share the same `gate_mm_args` and activation source, but the
  measured paired gate/up kernels were slower than two separate matmuls;
- the stage profile shows the `map` stage is not the target; early-window
  gate/up/down matmul time is.

Arithmetic/layout constraints for the next real kernel:

- The legacy `kernel_mul_mm_id` path uses a 64-row by 32-token tile, legacy
  threadgroup layout, `simdgroup_load`, and eight
  `simdgroup_multiply_accumulate` accumulators. This is the reference behavior
  for low-drift output order.
- The current fast-layout path changes the threadgroup tensor layout and uses
  Metal 4 cooperative tensors. It is fast, but widening it into early layers
  causes route-wide Tensor-vs-standard drift; local per-projection comparator
  deltas alone are not enough to prove promotion safety.
- A replacement should first preserve the legacy output layout and writeback
  order, then remove overhead around loads, barriers, or pointer/index setup.
  Starting from cooperative tensor math is acceptable only if the local
  comparator stays tight and the five-fixture gate remains green.

Prototype acceptance order:

1. Build and route the candidate behind a default-off env var.
2. Run a local comparator probe for the touched route (`moe_gate`, `moe_up`, or
   `moe_down`) with enough comparisons to cover early and late layers.
3. Run `run_prefill_candidate_gate.py` without drift first. The candidate must
   clear both the median and repeat-level compact prefill floors.
4. Only then run the five-fixture drift gate. Promotion still requires no new
   top-1 mismatch, no Tensor-vs-standard greedy mismatch, and Tensor-vs-standard
   worst RMS/top20 abs inside the configured envelope.

This rules out another small route-window probe as the next step. The next code
candidate should be a new routed-MoE matmul variant with an explicit comparator
route and speed-gate artifact.

## Rejected Q8_0 N64 Dense Tile

Artifact roots:

- `speed-bench/local-runs/20260514-215521-q8-n64-attn-q-b/`
- `speed-bench/local-runs/20260514-215814-q8-n64-attn-out/`

Patch tested: an experimental `kernel_mul_mm_q8_0_f32_n64` with 64 token
columns and eight simdgroups, guarded by `DS4_METAL_Q8_PREFILL_N64=1` plus an
optional route filter. The kernel preserved the legacy Q8_0 dequantization and
per-element accumulation order, but widened the token tile from 32 to 64.

Compact timing versus the current Tensor baseline was not a clean win:

| Candidate | 512 | 1024 | 2048 | 4096 | 8192 | Generation impact |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `attn_q_b` N64 | -4.4% | -1.6% | -0.9% | +0.2% | +0.9% | -2.0% to +0.7% |
| `attn_out` N64 | -4.8% | -2.2% | -0.3% | +0.1% | +0.8% | -0.7% to +0.6% |

Decision: revert/no production knob. The wider tile helped an isolated profile
stage in places, but whole-model compact prefill regressed short contexts and
only improved long contexts by less than 1%. This was rejected before running
the drift gate because the performance bar was not met.

## Dense Q8_0 Prototype Boundary

The current generic dense Q8_0 prefill dispatch is back on the legacy
`kernel_mul_mm_q8_0_f32` path: 64 output rows by 32 token columns, four
SIMD-group MMA slices for the output rows, and two SIMD-group MMA slices for
the token columns. It already uses `simdgroup_multiply_accumulate` and preserves
the legacy dequantization/reduction order.

Rejected or reverted dense Q8_0 directions now cover the obvious low-risk
scheduling variants:

- splitting full 32-token tiles from the tail was noise-level
  (`+0.3%` prefill on the targeted long fixture);
- widening the token tile to 64 (`kernel_mul_mm_q8_0_f32_n64`) was not a
  whole-model win;
- cooperative/direct-RHS Tensor prototypes for `attn_q_b` and `attn_output_b`
  either regressed mid-context/generation or failed the five-fixture gate.

Conclusion: do not add another dense Q8_0 switch without a genuinely new kernel
design. The next Q8_0 attempt should be a separate default-off kernel family
with its own comparator and five-fixture gate, not a small variant of the
current legacy wrapper.

## Cleaned Baseline Drift Gate

Artifact root:

- `speed-bench/local-runs/20260514-221837-quality-drift-gate/`

Command:

```sh
python3 speed-bench/run_quality_drift_gate.py
```

Result: gate OK after removing the rejected N64 source patch.

| Pair | top1 mismatches | greedy mismatches | min top20 | worst rms | worst top20 abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard vs quality | 0 | 1 | 18/20 | 0.618172 | 2.24006 |
| tensor vs quality | 0 | 1 | 18/20 | 0.618172 | 2.24006 |
| tensor vs standard | 0 | 0 | 19/20 | 0.239946 | 0.55422 |

Conclusion: the current conservative Tensor default remains drift-controlled
relative to standard Metal. The one greedy mismatch is already present in
standard Metal versus `--quality`; Tensor does not add a greedy mismatch against
standard in the five-fixture gate.

The same saved five-fixture dumps were later regenerated with the production
Tensor-vs-standard envelope enabled:

```sh
python3 speed-bench/run_quality_drift_gate.py \
  --reuse \
  --out-dir speed-bench/local-runs/20260514-221837-quality-drift-gate \
  --max-tensor-standard-rms 0.30 \
  --max-tensor-standard-top20-abs 0.60
```

Result: gate OK. Tensor-vs-standard remained at zero top-1 mismatches, zero
greedy mismatches, min top20 overlap `19/20`, worst RMS `0.239946`, and worst
top20 max abs `0.55422`, so the current conservative default is inside the
strict promotion envelope.

## Rejected FlashAttention Static Mask Cache

Artifact root:

- `speed-bench/local-runs/20260514-235636-flash-attn-mask-cache/`

Command:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --candidate-label flash-attn-mask-cache \
  --set-env DS4_METAL_FLASH_ATTN_MASK_CACHE=1 \
  --repeat 2 \
  --ctx-max 8192 \
  --gen-tokens 16
```

Patch tested: a default-off cache for static mixed FlashAttention prefill masks
and block maps, limited to the non-vector static mixed path.

Median timing versus the current Tensor baseline:

| ctx | candidate vs Tensor prefill | candidate vs Tensor generation |
| ---: | ---: | ---: |
| 512 | -3.9% | -1.3% |
| 1024 | -4.3% | -0.2% |
| 2048 | -2.4% | -0.3% |
| 4096 | -0.2% | -0.4% |
| 8192 | +1.2% | -0.0% |

Decision: revert/no production knob. The cache removes repeated mask/block-map
work in the stage profiler, but whole-model compact prefill regresses short and
mid contexts and only improves the 8192-token point by 1.2%. This was rejected
before running the drift gate because the performance bar was not met.

## Rejected FlashAttention CPU Block Map

Artifact root:

- `speed-bench/local-runs/20260515-000658-flash-attn-cpu-block-map/`

Command:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --candidate-label flash-attn-cpu-block-map \
  --set-env DS4_METAL_FLASH_ATTN_CPU_BLOCK_MAP=1 \
  --repeat 2 \
  --ctx-max 8192 \
  --gen-tokens 16
```

Patch tested: a default-off analytic CPU block-map fill for static mixed
non-vector FlashAttention prefill. The candidate used per-call transient block
buffers to avoid CPU writes racing later GPU reads in the shared command
buffer.

`DS4_METAL_FLASH_ATTN_CPU_BLOCK_MAP=1 ./ds4_test --metal-mpp-equivalence`
passed with the same summary as the current default:
`top1_mismatch=0`, `greedy_fail=0`, `worst_rms=0.239946`,
`worst_top20_max_abs=0.55422`.

Median timing versus the current Tensor baseline:

| ctx | candidate vs Tensor prefill | candidate vs Tensor generation |
| ---: | ---: | ---: |
| 512 | +2.3% | -0.1% |
| 1024 | -0.9% | -3.1% |
| 2048 | -3.1% | -2.7% |
| 4096 | +0.5% | +0.2% |
| 8192 | -0.3% | +0.0% |

Decision: revert/no production knob. Avoiding the GPU block-map dispatch is not
a stable whole-model win once the extra CPU work and transient buffer allocation
are included.

## Rejected FlashAttention NSG4 Geometry

Artifact root:

- `speed-bench/local-runs/20260515-001146-flash-attn-nsg4/`

Command:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --candidate-label flash-attn-nsg4 \
  --set-env DS4_METAL_FLASH_ATTN_NSG4=1 \
  --repeat 2 \
  --ctx-max 8192 \
  --gen-tokens 16
```

Patch tested: a host-only default-off switch that kept the existing non-vector
static mixed FlashAttention `Q=8,C=64` specialization but changed the runtime
simdgroup count from `NSG=8` to `NSG=4`, making each simdgroup handle two query
rows.

Median timing versus the current Tensor baseline:

| ctx | candidate vs Tensor prefill | candidate vs Tensor generation |
| ---: | ---: | ---: |
| 512 | -10.4% | -2.0% |
| 1024 | -6.8% | -1.0% |
| 2048 | -6.8% | -1.1% |
| 4096 | -4.2% | -0.9% |
| 8192 | -0.3% | -0.8% |

Decision: revert/no production knob. The lower simdgroup count consistently
regresses compact prefill and slightly hurts generation, so the default `NSG=8`
remains the right geometry for the current static mixed path.

## Q/KV RMS Fusion Boundary

Artifact root:

- `speed-bench/local-runs/20260515-001750-disable-qkv-norm-fusion/`

Command:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --candidate-label disable-qkv-norm-fusion \
  --set-env DS4_METAL_DISABLE_QKV_NORM_FUSION=1 \
  --repeat 2 \
  --ctx-max 8192 \
  --gen-tokens 16
```

Patch tested: no code change; this uses the existing reference-path switch to
disable the default fused Q/KV RMSNorm path in prefill.

Median timing versus the current Tensor baseline:

| ctx | disabled fusion vs Tensor prefill | disabled fusion vs Tensor generation |
| ---: | ---: | ---: |
| 512 | -5.1% | -2.5% |
| 1024 | -6.1% | -1.8% |
| 2048 | -4.2% | -2.0% |
| 4096 | -1.7% | -0.8% |
| 8192 | +1.4% | -1.3% |

Decision: keep the Q/KV RMSNorm fusion enabled by default. Disabling it is a
short/mid-context regression and hurts generation at every compact point.

## Compressor Pair Projection Scope

No benchmark run.

`DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ` and
`DS4_METAL_COMPRESSOR_PAIR_NR4` were inspected as possible compressor
projection boundaries. Both are decode-scoped in the current graph path:

- `DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ` selects the reference pair of F16
  matvecs instead of `ds4_gpu_matmul_f16_pair_tensor()` while updating
  compressed KV/indexer state for the current decode token.
- `DS4_METAL_COMPRESSOR_PAIR_NR4` only changes the paired F16 Tensor matvec
  dispatch when `n_tok == 1`.

Decision: skip them for prefill optimization. They may be useful for a focused
decode throughput A/B later, but they do not address compact prefill time.

## Rejected FlashAttention Q4 Geometry

Artifact root:

- `speed-bench/local-runs/20260515-002819-flash-attn-q4/`

Command:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --candidate-label flash-attn-q4 \
  --set-env DS4_METAL_FLASH_ATTN_Q4=1 \
  --repeat 2 \
  --ctx-max 8192 \
  --gen-tokens 16
```

Patch tested: a default-off non-vector static-mixed FlashAttention
specialization with `Q=4,C=64,NSG=4`, compared with the current
`Q=8,C=64,NSG=8` default.

Median timing versus the current Tensor baseline:

| ctx | candidate vs Tensor prefill | candidate vs Tensor generation |
| ---: | ---: | ---: |
| 512 | -11.3% | -1.0% |
| 1024 | -2.7% | -0.5% |
| 2048 | -0.7% | +0.3% |
| 4096 | +0.7% | -0.2% |
| 8192 | +0.9% | -2.4% |

Decision: revert/no production knob and no drift gate. Smaller query tiles
hurt short-context compact prefill and only give sub-1% long-context gains,
with a generation regression at 8192.

## RMSNorm Rsqrt Boundary

Artifact root:

- `speed-bench/local-runs/20260515-003403-norm-rsqrt/`

Command:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --candidate-label norm-rsqrt \
  --set-env DS4_METAL_NORM_RSQRT_DISABLE=0 \
  --repeat 2 \
  --ctx-max 8192 \
  --gen-tokens 16
```

Patch tested: no code change; this disables the current drift-stabilizing
RMSNorm unification macro and restores hardware `rsqrt()` in
`kernel_rms_norm_f32`.

Median timing versus the current Tensor baseline:

| ctx | `rsqrt()` vs Tensor prefill | `rsqrt()` vs Tensor generation |
| ---: | ---: | ---: |
| 512 | -1.8% | +0.2% |
| 1024 | -3.7% | -0.4% |
| 2048 | -2.7% | -0.5% |
| 4096 | -2.5% | -0.6% |
| 8192 | -0.9% | -0.9% |

Decision: keep `DS4_METAL_NORM_RSQRT_DISABLE` enabled by default. Restoring
hardware `rsqrt()` is slower at every compact prefill point and would also
remove a deliberate drift-control patch, so no drift gate was run.

## Prefill Chunk Size Boundary

Artifact root:

- `speed-bench/local-runs/20260515-003739-prefill-chunk-full/`

Command:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --candidate-label prefill-chunk-full \
  --set-env DS4_METAL_PREFILL_CHUNK=0 \
  --repeat 2 \
  --ctx-max 8192 \
  --gen-tokens 16
```

Patch tested: no code change; this uses the existing `DS4_METAL_PREFILL_CHUNK=0`
override to prefill each prompt as one full chunk instead of using the default
4096-token cap for long prompts.

Median timing versus the current Tensor baseline:

| ctx | full chunk vs Tensor prefill | full chunk vs Tensor generation |
| ---: | ---: | ---: |
| 512 | -7.3% | -0.1% |
| 1024 | -1.2% | -0.2% |
| 2048 | -1.8% | -1.1% |
| 4096 | -3.3% | -2.0% |
| 8192 | -1.0% | -0.4% |

Decision: keep the default 4096-token long-prompt prefill cap. Full-prompt
prefill was slower at every compact point, so no drift gate was run.

The smaller `DS4_METAL_PREFILL_CHUNK=2048` cap was also screened later:

- `speed-bench/local-runs/20260515-051759-prefill-chunk-2048-screen/prefill-candidate-summary.md`

One-repeat timing versus the current Tensor baseline:

| ctx | 2048 chunk vs Tensor prefill | 2048 chunk vs Tensor generation |
| ---: | ---: | ---: |
| 512 | +0.1% | -1.0% |
| 1024 | -1.4% | -0.9% |
| 2048 | +0.7% | -0.1% |
| 4096 | +1.6% | -1.0% |
| 8192 | -7.0% | -4.5% |

Decision: reject before drift. Smaller chunks give a small 2048/4096 bump in
this noisy single-repeat screen but regress the 8192 point badly and increase
dispatch/setup pressure. Keep the default 4096-token cap for compact and
long-context prefill timing.

The larger `DS4_METAL_PREFILL_CHUNK=8192` cap was screened later with the
current strict two-repeat candidate gate:

- `speed-bench/local-runs/20260515-170138-prefill-chunk-8192-screen/prefill-candidate-summary.md`

Two-repeat median timing versus the current Tensor baseline:

| ctx | 8192 chunk vs Tensor prefill | 8192 chunk vs Tensor generation |
| ---: | ---: | ---: |
| 512 | -8.2% | -0.4% |
| 1024 | -3.6% | +1.7% |
| 2048 | -1.7% | -0.7% |
| 4096 | -0.5% | -1.2% |
| 8192 | +1.4% | -0.8% |

Decision: reject before drift. The median line only helps at 8192 tokens, and
the repeat-level prefill floor was much worse (`-12.1%`). This closes the
obvious chunk-size boundary: `2048`, full-prompt, and `8192` chunks all lose to
the default 4096-token cap under the compact speed screen.

Refreshed local run index after this artifact:

- `speed-bench/local-runs/20260515-170446-local-run-index/local-run-index.md`

## Rejected RoPE exp2/log2 Arithmetic

Artifact root:

- `speed-bench/local-runs/20260515-004221-rope-exp2-log2/`

Command:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --candidate-label rope-exp2-log2 \
  --set-env DS4_METAL_ROPE_EXP2_LOG2=1 \
  --repeat 2 \
  --ctx-max 8192 \
  --gen-tokens 16
```

Patch tested: no code change; this uses the existing diagnostic macro that
computes RoPE frequency powers as `exp2(log2())` instead of `pow()`.

Median timing versus the current Tensor baseline:

| ctx | exp2/log2 vs Tensor prefill | exp2/log2 vs Tensor generation |
| ---: | ---: | ---: |
| 512 | -0.8% | -0.4% |
| 1024 | -0.5% | -0.5% |
| 2048 | -1.2% | -0.8% |
| 4096 | -1.9% | -0.3% |
| 8192 | -1.5% | -1.2% |

Decision: keep the default `pow()` RoPE path. The `exp2(log2())` variant is
slower at every compact prefill point and also slightly hurts generation, so no
drift gate was run.

## KV Raw F32 Precision Boundary

Artifact root:

- `speed-bench/local-runs/20260515-004510-kv-raw-f32/`

Command:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --candidate-label kv-raw-f32 \
  --set-env DS4_METAL_KV_RAW_F32=1 \
  --repeat 2 \
  --ctx-max 8192 \
  --gen-tokens 16
```

Patch tested: no code change; this uses the existing diagnostic macro that
keeps raw KV cache values in F32 instead of matching the half-typed
FlashAttention KV buffer precision.

Median timing versus the current Tensor baseline:

| ctx | F32 raw KV vs Tensor prefill | F32 raw KV vs Tensor generation |
| ---: | ---: | ---: |
| 512 | +0.2% | +0.5% |
| 1024 | -0.0% | -0.6% |
| 2048 | +1.1% | +0.1% |
| 4096 | +0.2% | -0.5% |
| 8192 | -0.2% | -0.4% |

Decision: keep F32 raw KV default-off. The compact speed result is noise-level
and mixed, while the macro intentionally changes a precision boundary between
the raw indexer view and the FlashAttention half KV view. No drift gate was run.

## Routed-MoE Gate/Up Disable Boundary

Artifact root:

- `speed-bench/local-runs/20260515-005052-moe-gate-up-disable/`

Command:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --candidate-label moe-gate-up-disable \
  --set-env DS4_METAL_MPP_MOE_GATE_DISABLE=1 \
  --set-env DS4_METAL_MPP_MOE_UP_DISABLE=1 \
  --repeat 2 \
  --ctx-max 8192 \
  --gen-tokens 16
```

Patch tested: no code change; this disables only the current routed-MoE gate
and up Tensor routes while leaving the promoted down route enabled.

Median timing versus the current Tensor baseline:

| ctx | disabled gate/up vs Tensor prefill | disabled gate/up vs Tensor generation |
| ---: | ---: | ---: |
| 512 | -19.5% | -0.6% |
| 1024 | -21.4% | -0.0% |
| 2048 | -18.5% | +0.1% |
| 4096 | -13.9% | -0.1% |
| 8192 | -9.7% | -0.1% |

Decision: keep the current gate/up Tensor window enabled. Disabling those
routes removes a large part of the compact prefill win, so no drift gate was
run.

## Routed-MoE Down Disable Boundary

Artifact root:

- `speed-bench/local-runs/20260515-005523-moe-down-disable/`

Command:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --candidate-label moe-down-disable \
  --set-env DS4_METAL_MPP_MOE_DOWN_DISABLE=1 \
  --repeat 2 \
  --ctx-max 8192 \
  --gen-tokens 16
```

Patch tested: no code change; this disables only the current routed-MoE down
Tensor route while keeping the promoted gate/up routes enabled.

Median timing versus the current Tensor baseline:

| ctx | disabled down vs Tensor prefill | disabled down vs Tensor generation |
| ---: | ---: | ---: |
| 512 | -10.1% | -0.4% |
| 1024 | -12.5% | -1.1% |
| 2048 | -10.0% | -0.1% |
| 4096 | -7.3% | +0.5% |
| 8192 | -5.8% | +0.4% |

Decision: keep the current down Tensor window enabled. Disabling the down route
also removes a clear compact prefill win, so no drift gate was run.

## GPU Embedding Threshold Boundary

Artifact root:

- `speed-bench/local-runs/20260515-010001-gpu-embed-min2048/`

Command:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --candidate-label gpu-embed-min2048 \
  --set-env DS4_METAL_GPU_BATCH_EMBED_MIN=2048 \
  --repeat 2 \
  --ctx-max 8192 \
  --gen-tokens 16
```

Patch tested: no code change; this raises the batched prompt embedding GPU
crossover from 512 tokens to 2048 tokens, forcing the 512- and 1024-token
compact points through the CPU embedding upload path.

Median timing versus the current Tensor baseline:

| ctx | threshold 2048 vs Tensor prefill | threshold 2048 vs Tensor generation |
| ---: | ---: | ---: |
| 512 | -0.7% | +0.4% |
| 1024 | -1.3% | +0.4% |
| 2048 | -1.7% | -1.0% |
| 4096 | -4.0% | -1.0% |
| 8192 | -1.0% | -0.5% |

Decision: keep the default 512-token GPU embedding crossover. Raising the
threshold did not help the short contexts and regressed the whole compact
sweep, so no drift gate was run.

## Boundary Sweep Conclusion

The current env-only and low-risk patch search has covered the production
prefill routes that are still relevant on this branch:

- routed-MoE Tensor defaults are independently justified: disabling gate/up or
  down regresses compact prefill by 5.8% to 21.4%;
- attention-output Tensor low projection is justified and its known tile/direct
  RHS alternatives have been rejected;
- F16 compressor Tensor default is justified, while pair/wide variants are
  either slower or drift-prone;
- dense Q8_0 and FlashAttention tile/setup variants have been rejected unless a
  genuinely new kernel design is introduced;
- precision/math boundaries (`rsqrt`, RoPE `exp2/log2`, F32 raw KV) do not
  provide useful prefill speed and are not promotion candidates;
- prefill scheduling/setup boundaries (`DS4_METAL_PREFILL_CHUNK=0`,
  `DS4_METAL_GPU_BATCH_EMBED_MIN=2048`) are slower than the current defaults.

Remaining untested switches are not good prefill optimization candidates:

- `DS4_METAL_NO_PREFILL_KERNEL_WARMUP`, `DS4_METAL_NO_MODEL_WARMUP`,
  `DS4_METAL_NO_RESIDENCY`, and
  `DS4_METAL_DISABLE_HOT_PIPELINE_STATICS` change startup/warmup behavior, not
  steady-state prefill kernel throughput.
- `DS4_METAL_DISABLE_COMPRESSOR_STORE_ONE`,
  `DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ`,
  `DS4_METAL_COMPRESSOR_PAIR_NR4`, `DS4_METAL_INDEXED_ATTN_RB4`,
  `DS4_METAL_DECODE_INDEXER_*`, and the fused decode `DS4_METAL_DISABLE_*`
  switches are decode-scoped for this compact prefill gate.
- `DS4_METAL_TENSOR_MATMUL_DISABLE=1`, `DS4_METAL_TENSOR_DISABLE=1`, and
  `DS4_METAL_MPP_DISABLE=1` are global negative controls that collapse the
  current promoted Tensor routes back toward the standard Metal baseline; the
  route-specific disable checks above provide more actionable evidence.

Next useful optimization work should therefore be code-design work rather than
another env sweep:

1. a new routed-MoE matmul design that preserves the fast all-layer profile
   while reducing Tensor-vs-standard drift;
2. a genuinely new dense Q8_0 prefill kernel family for `attn_q_b` or
   `attn_output_b`, with its own comparator and five-fixture gate;
3. a real static-mixed FlashAttention kernel redesign rather than changing
   only query/key tile sizes or setup kernels.

Promotion rule remains unchanged: keep a change only if compact prefill timing
improves and the five-fixture gate shows no new top-1 mismatch and no new
Tensor-vs-standard greedy continuation mismatch.

## Routed-MoE Kernel Design Triage

Code inspection of the current routed-MoE prefill path confirms there is not an
obvious one-line drift fix left in the existing Tensor route. The host selector
uses the fast MPP layout by default for routed-MoE unless `N=64` tiles or
`DS4_METAL_MPP_MOE_FAST_LAYOUT=0` are requested. Both the generic MPP variant
and the fast layout variant ultimately accumulate through Metal 4
`matmul2d::run(...)`; the non-MPP reference in the same template keeps the
legacy `simdgroup_multiply_accumulate` loop and is what the route comparator
replays for local checks.

That matches the measurements: disabling fast layout, widening to 64-token
tiles, pairing gate/up, and forcing F32 mid storage either regressed speed or
did not reduce the full-model Tensor-vs-standard drift. Comparator scans found
actionable local `moe_down` outliers at the already-skipped layers, while
gate/up did not show a single large local breach. The remaining movement is
therefore accumulated route-wide arithmetic movement from the cooperative Tensor
matmul, not a small dispatch or precision-boundary bug.

Next routed-MoE work should be a new default-off kernel family with a comparator
from day one. The remaining useful direction is a reference-order simdgroup
kernel that preserves the legacy reduction shape but improves expert-major
staging and writeback around the prefill map.

The later skip-26/29/30 and clean-early hybrid probes already tested the
selective `moe_down` idea: local comparator exclusions reduced the largest
projection outliers, but the full five-fixture Tensor-vs-standard envelope still
failed. Treat further route-filtering as exhausted unless a new kernel changes
the local arithmetic or output layout first.

Do not promote another route-window change unless it improves compact prefill
and passes the five-fixture gate with no new top-1 mismatch and no new
Tensor-vs-standard greedy continuation mismatch.

## Drift Gate Artifact Update

`speed-bench/run_quality_drift_gate.py` now writes `summary.md` beside
`summary.json`. The Markdown report contains the same five-scenario tables for
`standard_vs_quality`, `tensor_vs_quality`, and `tensor_vs_standard`, plus the
aggregate gate status. This keeps the promotion evidence persistent and
human-readable under the ignored `speed-bench/local-runs/` artifact tree.

Validation used the existing current-default drift dumps with `--reuse`:

```sh
python3 speed-bench/run_quality_drift_gate.py \
  --reuse \
  --out-dir speed-bench/local-runs/20260514-221837-quality-drift-gate
```

The regenerated Markdown report is:

- `speed-bench/local-runs/20260514-221837-quality-drift-gate/summary.md`

Gate result stayed `OK`: Tensor-vs-standard had zero top-1 mismatches, zero
greedy mismatches, min top20 overlap `19/20`, worst RMS `0.239946`, and worst
top20 max abs `0.55422`.

`speed-bench/run_prefill_candidate_gate.py` now also writes
`prefill-candidate-summary.md` beside `prefill-candidate-summary.json`. The
candidate Markdown report combines the median compact speed table with the
five-scenario drift-gate status when `--run-drift-gate` is used and the speed
screen passes. If the speed screen fails or the drift gate is otherwise not
run, the report says so explicitly to avoid promoting speed-only candidate
artifacts.

The candidate scorecard also computes a conservative promotion decision:

- every measured compact context must beat the Tensor baseline by at least
  `--min-prefill-gain-pct` (default `0.0`);
- every repeat/context pair must clear `--min-repeat-prefill-gain-pct`
  (default `0.0`), and the Markdown report now prints the per-context repeat
  deltas so median-only wins are easy to audit;
- the five-scenario drift gate must be present and green;
- Tensor-vs-standard drift must stay inside the configured production envelope:
  `--max-tensor-standard-rms=0.30` and
  `--max-tensor-standard-top20-abs=0.60` by default;
- failed speed screens skip the nested drift gate and still write
  JSON/Markdown artifacts; failed drift gates also write artifacts before
  returning non-zero. Pass `--no-fail` for exploratory sweeps that should keep
  going after a rejected candidate.

Writer validation used the existing `gpu-embed-min2048` candidate summary
without rerunning benchmarks:

- `speed-bench/local-runs/20260515-010001-gpu-embed-min2048/prefill-candidate-summary.md`

`--reuse --out-dir=<existing-run>` now regenerates candidate scorecards from
saved CSVs/charts and passes `--reuse` through to nested drift-gate dumps. This
was validated on the default-off fast routed-MoE skip candidate without
rerunning benchmarks or model captures:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --reuse \
  --out-dir speed-bench/local-runs/20260514-212340-mpp-fast-skip-down26-29-30 \
  --candidate-label mpp-fast-skip-down26-29-30 \
  --set-env DS4_METAL_MPP_FAST=1 \
  --set-env DS4_METAL_MPP_MOE_DOWN_FILTER=layer=0-25,layer=27-28,layer=31-42 \
  --repeat 2 \
  --ctx-max 8192 \
  --gen-tokens 16 \
  --run-drift-gate \
  --no-fail
```

The regenerated scorecard correctly reports that the candidate is not
production promotion-safe under the default drift envelope even though it is a
useful default-off eval candidate: it passes top-1/greedy gates and has minimum
compact prefill gain `+6.0%`, but Tensor-vs-standard worst RMS `0.64381` and
worst top20 abs `1.13945` exceed the production envelope.

The standalone `run_quality_drift_gate.py` also accepts the same optional drift
envelope flags. The candidate gate passes them through to the nested drift gate,
so the nested `quality-drift-gate/summary.md` now reports `Gate: FAIL` for
production-envelope breaches while still preserving the raw five-scenario
tables.

## Stage Profile Shape Tables

`speed-bench/summarize_stage_profile.py` now keeps per-shape totals for dense
Q8_0 profile lines, matching the existing FlashAttention shape tables. This
makes the dense matmul targets explicit in persistent local reports instead of
requiring manual parsing of stderr.

Validation regenerated a summary from the existing current-default profile log
without rerunning benchmarks:

```sh
python3 speed-bench/summarize_stage_profile.py \
  speed-bench/local-runs/20260514-161802-current-default-attn-all-profile/long_code_audit_profile.log \
  --output speed-bench/local-runs/20260515-012815-stage-profile-summary/stage-profile-summary.md \
  --json speed-bench/local-runs/20260515-012815-stage-profile-summary/stage-profile-summary.json
```

The generated Q8 shape table ranks `attn_out in=8192 out=4096 tok=3844` at
`808.055 ms` total and `attn_q_b in=1024 out=32768 tok=3844` at `805.319 ms`
total, followed by `attn_q_a` and `attn_kv`. These ignored local artifacts are
kept under:

- `speed-bench/local-runs/20260515-012815-stage-profile-summary/stage-profile-summary.md`
- `speed-bench/local-runs/20260515-012815-stage-profile-summary/stage-profile-summary.json`

## Candidate Generation Floor

`speed-bench/run_prefill_candidate_gate.py` now treats generation throughput as
a secondary promotion condition instead of an informational-only column. The
scorecard still prioritizes prefill, but a candidate is not production-safe if
any measured context falls below `--min-generation-gain-pct` versus the current
Tensor baseline. The default floor is `-5.0%`, which allows small generation
noise for prefill-first work while rejecting larger regressions before eval.

Negative-control validation reused the saved long-context CSVs for
`mpp-fast-gate0-up15-down12-long128` without rerunning benchmarks:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --reuse \
  --out-dir speed-bench/local-runs/20260514-191816-mpp-fast-gate0-up15-down12-long128 \
  --candidate-label mpp-fast-gate0-up15-down12-long128 \
  --set-env DS4_METAL_MPP_FAST=1 \
  --set-env DS4_METAL_MPP_MOE_UP_START_LAYER=15 \
  --set-env DS4_METAL_MPP_MOE_DOWN_START_LAYER=12 \
  --repeat 1 \
  --ctx-max 65536 \
  --gen-tokens 128 \
  --no-fail
```

The regenerated scorecard fails promotion for both the prefill floor
(`min=-3.9%`) and the generation floor (`min=-8.0%`, required `-5.0%`), and
also notes that the drift gate was not run:

- `speed-bench/local-runs/20260514-191816-mpp-fast-gate0-up15-down12-long128/prefill-candidate-summary.md`

The candidate gate also now records repeat-level prefill gains and requires
every repeat/context pair to clear `--min-repeat-prefill-gain-pct` before
marking a candidate promotion-safe. The default is `0.0%`, matching the median
prefill floor but avoiding hidden one-repeat regressions in noisy two-repeat
screens. Repeat-level generation is reported as a diagnostic, while the
promotion floor for generation remains median-based because short generation
timing is noisier than prefill timing.

## Drift Worst-Fixture Attribution

`speed-bench/run_quality_drift_gate.py` now writes an `extrema` block for each
pair and adds a "Worst fixture" table to `summary.md`. Drift-envelope failures
also name the fixture that caused the breach.

Validation regenerated the existing fast skip-26/29/30 drift summary with
`--reuse`, without rerunning logits or logprobs captures:

```sh
python3 speed-bench/run_quality_drift_gate.py \
  --reuse \
  --out-dir speed-bench/local-runs/20260514-212340-mpp-fast-skip-down26-29-30/quality-drift-gate \
  --max-tensor-standard-rms 0.30 \
  --max-tensor-standard-top20-abs 0.60 \
  --set-env DS4_METAL_MPP_FAST=1 \
  --set-env DS4_METAL_MPP_MOE_DOWN_FILTER=layer=0-25,layer=27-28,layer=31-42 \
  --no-fail
```

For `tensor_vs_standard`, the envelope failures are now attributed to
`long_memory_archive` for worst RMS (`0.64381`) and `long_code_audit` for worst
top20 abs (`1.13945`). The parent prefill candidate scorecard was regenerated
from saved CSVs and now carries those fixture names in its promotion failures
and its compact drift-target table:

- `speed-bench/local-runs/20260514-212340-mpp-fast-skip-down26-29-30/quality-drift-gate/summary.md`
- `speed-bench/local-runs/20260514-212340-mpp-fast-skip-down26-29-30/prefill-candidate-summary.md`

Both `run_quality_drift_gate.py` and `run_prefill_candidate_gate.py` now write a
`run_config` JSON block, and their Markdown reports show a compact Run Config
table. This preserves the thresholds, context range, repeat count, reuse mode,
resolved tool paths, and command arguments needed to reproduce a saved baseline
or candidate gate. The Markdown reports also include a quoted replay command so
the same gate can be copied directly into a shell.

## Persistent Local Artifacts

`speed-bench/run_metal_tensor_bench.sh` now defaults to a timestamped ignored
output directory:

```sh
OPEN_CHART=0 speed-bench/run_metal_tensor_bench.sh
```

The current branch chart was regenerated and kept locally at:

- `speed-bench/local-runs/20260514-220230-metal-tensor-bench/ds4_bench_standard_quality_tensor_128.png`
- `speed-bench/local-runs/20260515-021428-metal-tensor-bench/ds4_bench_standard_quality_tensor_128.png`

`speed-bench/index_local_runs.py` builds a persistent Markdown/JSON index across
saved local run summaries without rerunning benchmarks or drift captures:

```sh
RUN_ID=$(date +%Y%m%d-%H%M%S)
OUT_DIR=speed-bench/local-runs/${RUN_ID}-local-run-index
python3 speed-bench/index_local_runs.py \
  --output ${OUT_DIR}/local-run-index.md \
  --json-output ${OUT_DIR}/local-run-index.json
```

Validation artifact:

- `speed-bench/local-runs/20260515-015819-local-run-index/local-run-index.md`

Refreshed local index after the comparator follow-up:

- `speed-bench/local-runs/20260515-021401-local-run-index/local-run-index.md`

Refreshed local index after the full current-branch chart regeneration:

- `speed-bench/local-runs/20260515-022807-local-run-index/local-run-index.md`

Refreshed local index after the gate/up-fast, down-clean-early hybrid rejection:

- `speed-bench/local-runs/20260515-023724-local-run-index/local-run-index.md`

Refreshed local index after the dense Q8_0 comparator smoke:

- `speed-bench/local-runs/20260515-024233-local-run-index/local-run-index.md`

Refreshed local index after wiring Q8 into the comparator probe wrapper:

- `speed-bench/local-runs/20260515-024511-local-run-index/local-run-index.md`

Refreshed local index after adding `q8_filter` to the comparator probe run
config:

- `speed-bench/local-runs/20260515-024648-local-run-index/local-run-index.md`

Refreshed local index after the `attn_out` dense Q8_0 comparator smoke:

- `speed-bench/local-runs/20260515-024755-local-run-index/local-run-index.md`

Refreshed local index after the long-shape dense Q8_0 comparator baselines:

- `speed-bench/local-runs/20260515-025020-local-run-index/local-run-index.md`

## Comparator Continue-On-Breach Probe

The local comparator can now keep scanning after a target breach:

```sh
python3 speed-bench/run_mpp_compare_probe.py \
  --preset mpp-fast-skip-down26-29-30 \
  --case long_memory_archive \
  --route moe_down \
  --continue-after-breach \
  --compare-max 80 \
  --top 12
```

Validation artifact:

- `speed-bench/local-runs/20260515-021315-mpp-fast-skip-down26-29-30-mpp-compare-probe/mpp-compare-summary.md`

This confirms the rejected skip-26/29/30 candidate is not only a single
layer-31 local-delta issue. With continue-on-breach enabled, `moe_down`
breaches repeated across layers 31-40 and 42 on `long_memory_archive`; worst
local max abs was `0.0205078` at layer 42. This keeps the candidate rejected
and makes further down-projection expansion unattractive without a different
accuracy strategy.

## Dense Q8_0 Comparator Hook

Added a default-off dense Q8_0 comparator hook for future kernel prototypes:

```sh
DS4_METAL_Q8_COMPARE=1 \
DS4_METAL_Q8_COMPARE_FILTER=attn_q_b \
DS4_METAL_MPP_COMPARE_MAX=3 \
DS4_METAL_MPP_COMPARE_VERBOSE=1 \
./ds4 --metal -mt auto \
  --prompt-file tests/test-vectors/prompts/short_code_completion.txt \
  -c 4096 -n 1 --system "" --nothink --temp 0
```

Validation artifact:

- `speed-bench/local-runs/20260515-024144-q8-compare-smoke/mpp-compare-summary.md`

The smoke run compared the current legacy Q8_0 prefill output against a legacy
reference for the first three `attn_q_b` layers and reported zero delta for all
three `32768x27x1024` comparisons. This does not change production behavior or
promote a new kernel; it gives the next dense Q8_0 prototype a local
ref-vs-candidate check before the five-fixture logprob gate.

`speed-bench/run_mpp_compare_probe.py` now supports the same hook directly:

```sh
python3 speed-bench/run_mpp_compare_probe.py \
  --case short_code_completion \
  --route q8 \
  --q8-filter attn_q_b \
  --compare-max 3 \
  --verbose \
  --top 10
```

Validation artifact:

- `speed-bench/local-runs/20260515-024453-manual-mpp-compare-probe/mpp-compare-summary.md`
- `speed-bench/local-runs/20260515-024637-manual-mpp-compare-probe/mpp-compare-summary.md`

The wrapper set `DS4_METAL_Q8_COMPARE=1` and
`DS4_METAL_Q8_COMPARE_FILTER=attn_q_b`, then produced the same zero-delta
three-layer `attn_q_b` summary. Future Q8 kernel candidates can use this
wrapper instead of hand-written env commands before the five-fixture gate. The
newer artifact also records `q8_filter=attn_q_b` explicitly in `run_config`.

The second dense Q8_0 hotspot was smoke-checked through the same wrapper:

```sh
python3 speed-bench/run_mpp_compare_probe.py \
  --case short_code_completion \
  --route q8 \
  --q8-filter attn_out \
  --compare-max 3 \
  --verbose \
  --top 10
```

Validation artifact:

- `speed-bench/local-runs/20260515-024740-manual-mpp-compare-probe/mpp-compare-summary.md`

This produced three zero-delta `attn_out` comparisons with shape
`4096x27x8192`. Dense Q8_0 prototypes for both current hotspots now have a
one-command local comparator smoke before compact timing and the five-fixture
logprob gate.

Long-shape comparator baselines were also captured on `long_code_audit` with
`--compare-max 50 --verbose`, covering all 43 layers for each hotspot:

- `speed-bench/local-runs/20260515-024918-manual-mpp-compare-probe/mpp-compare-summary.md`
  (`attn_q_b`, 43 comparisons, shape `32768x3844x1024`, zero delta)
- `speed-bench/local-runs/20260515-024956-manual-mpp-compare-probe/mpp-compare-summary.md`
  (`attn_out`, 43 comparisons, shape `4096x3844x8192`, zero delta)

These are reference artifacts for the next dense Q8_0 kernel attempt. A useful
prototype should improve compact prefill timing, keep these local comparisons
inside target, then pass the five-fixture logprob gate before promotion.

## Current Default Baseline Refresh

Regenerated the full current-branch standard/quality/Tensor chart with
timestamped local artifacts:

```sh
OPEN_CHART=0 speed-bench/run_metal_tensor_bench.sh
```

Artifact root:

- `speed-bench/local-runs/20260515-025303-metal-tensor-bench/`

Chart:

- `speed-bench/local-runs/20260515-025303-metal-tensor-bench/20260515-025303_gen128_ds4_bench_standard_quality_tensor.png`

The Tensor default remains a clear prefill win over standard Metal on the full
512..65536 context sweep:

| ctx | Tensor prefill vs standard | Tensor generation vs standard |
| ---: | ---: | ---: |
| 512 | +31.3% | -0.9% |
| 1024 | +31.4% | -1.2% |
| 2048 | +26.5% | -0.7% |
| 4096 | +22.1% | -0.5% |
| 8192 | +19.9% | -0.8% |
| 16384 | +19.8% | -0.5% |
| 32768 | +16.6% | -0.6% |
| 65536 | +15.4% | -1.1% |

Also reran the strict five-fixture drift gate against the current source:

```sh
python3 speed-bench/run_quality_drift_gate.py \
  --max-tensor-standard-rms 0.30 \
  --max-tensor-standard-top20-abs 0.60
```

Artifact root:

- `speed-bench/local-runs/20260515-030753-quality-drift-gate/`

Result: `Gate: OK`.

Tensor-vs-standard stayed inside the conservative drift envelope:

| Metric | Value |
| --- | ---: |
| top1 mismatches | 0 |
| greedy mismatches | 0 |
| min top20 overlap | 19/20 |
| worst RMS | 0.239946 |
| worst top20 max abs | 0.55422 |

This is the current production baseline for the next prefill attempt: any new
default candidate should improve compact/full-sweep prefill while preserving a
green five-fixture gate and staying inside the `0.30` RMS / `0.60` top20
Tensor-vs-standard envelope.

## Current Stage Profile Refresh

Ran a fresh current-branch profile on `long_code_audit` with routed-MoE, dense
Q8_0, FlashAttention, and layer profiling enabled:

```sh
env DS4_METAL_MOE_STAGE_PROFILE=1 \
    DS4_METAL_Q8_PREFILL_PROFILE=1 \
    DS4_METAL_Q8_PREFILL_PROFILE_FILTER=attn_ \
    DS4_METAL_FLASH_ATTN_STAGE_PROFILE=1 \
    DS4_METAL_LAYER_PROFILE=1 \
    ./ds4 --metal -mt auto \
      --prompt-file tests/test-vectors/prompts/long_code_audit.txt \
      -c 16384 -n 1 --system "" --nothink --temp 0
```

Artifact root:

- `speed-bench/local-runs/20260515-031301-current-stage-profile/`

Summary:

- `speed-bench/local-runs/20260515-031301-current-stage-profile/stage-profile-summary.md`

The refreshed profile produced `420.69` prefill t/s and parsed `5001.333 ms`
of profiled stage time. The top stage families are still routed-MoE matmuls and
the two large dense Q8_0 attention projections:

| Stage | total ms | events | avg ms |
| --- | ---: | ---: | ---: |
| `moe_stage.gate` | 906.862 | 43 | 21.090 |
| `moe_stage.up` | 906.022 | 43 | 21.070 |
| `moe_stage.down` | 834.385 | 43 | 19.404 |
| `q8.attn_out` | 806.859 | 43 | 18.764 |
| `q8.attn_q_b` | 795.933 | 43 | 18.510 |
| `flash_attn.static_mixed_nonvec.attention` | 310.296 | 20 | 15.515 |

`speed-bench/summarize_stage_profile.py` now also reports routed-MoE timing by
Tensor mask. On this run:

| MoE mpp mask | top stages | total ms |
| --- | --- | ---: |
| `0/0/0` | `up`=410.4, `gate`=409.9, `down`=408.7 | 1266.616 |
| `1/1/1` | `gate`=397.5, `up`=395.3, `down`=385.3 | 1252.849 |
| `0/0/1` | `up`=100.4, `gate`=99.5, `down`=40.3 | 248.163 |

This makes the next prefill target concrete: a new routed-MoE kernel should
focus on the early legacy `0/0/0` window first. Simply switching those layers
to the existing cooperative-Tensor path has already been rejected by drift
gates, so the useful work is a reference-compatible MoE matmul design that
keeps the low-drift arithmetic behavior while reducing the early-window cost.
Dense Q8_0 `attn_out` and `attn_q_b` remain the next largest targets, but their
small tile/direct-RHS variants have already been rejected.

Legacy `kernel_mul_mm_id` inspection notes:

- the early `0/0/0` path already uses the same simdgroup MMA shape as the
  standard Metal reference;
- each expert-major tile produces a logical `64 x 32` result, but the 32
  columns map back through `hids` to token/expert slots rather than to a
  contiguous dense destination;
- the current threadgroup writeback is therefore doing a real scatter
  transpose, not just an avoidable staging copy;
- a useful reference-compatible kernel is more likely to improve expert-major
  staging or produce a token-major/down-sum layout directly than to replace the
  final scatter with a dense-style `simdgroup_store`.

That rules out the simplest "direct store" tweak. The next kernel prototype
should either change the work map/output layout deliberately or focus on
computing the routed down projection closer to the token-major summed output,
with a comparator before any timing gate.

## FlashAttention Vector-Path Boundary

The current static-mixed prefill router keeps the vector FlashAttention helper
only for `n_tokens < 20`; larger prefill batches use the non-vector helper. This
is not an arbitrary threshold. The vector helper launches `n_tokens * n_head *
nwg` workgroups and stores one partial `head_dim` result plus softmax state per
query/head/workgroup before a reduce pass:

```c
tmp_bytes = nrows * head_dim * nwg * sizeof(float) +
            nrows * (2 * nwg) * sizeof(float);
```

With the current DS4 shape (`n_head=64`, `head_dim=512`, `nwg=32`), forcing the
existing vector path for normal prefill would require the following temporary
buffer sizes:

| tokens | vector tmp |
| ---: | ---: |
| 16 | 64.2 MiB |
| 20 | 80.3 MiB |
| 64 | 257.0 MiB |
| 128 | 514.0 MiB |
| 256 | 1028.0 MiB |
| 512 | 2056.0 MiB |
| 1024 | 4112.0 MiB |
| 2048 | 8224.0 MiB |
| 4096 | 16448.0 MiB |
| 8192 | 32896.0 MiB |

Conclusion: reject a simple force-vector prefill patch before timing or drift.
The memory footprint is already about 2.0 GiB at 512 tokens and about 32.1 GiB
at 8192 tokens. Future FlashAttention prefill work needs a streaming or
reduced-temporary design; reusing the decode-style vector helper is not a
production candidate for normal prefill.

## Rejected M5 SIMD-Group Barrier Elision Probe

Checked the `swival-ds4-m5/simdgroup_matrix` idea of dropping the three
`simdgroup_barrier(mem_none)` calls inside the existing dense and routed-MoE
`simdgroup_multiply_accumulate` loops behind an M5 function constant. This
keeps the same MMA arithmetic, so it was a plausible low-drift prefill
candidate, but the timing was not favorable.

The local patch was tested and then reverted. The run used the candidate gate
in inverted form: `tensor` was the patched default-on M5 path, and
`disable-m5-sgmatrix-control` set `DS4_METAL_DISABLE_M5_SIMDGROUP_MATRIX=1`.

Artifact:

- `speed-bench/local-runs/20260515-032257-disable-m5-sgmatrix-control/prefill-candidate-summary.md`

Disabled control vs patched default:

| ctx | disabled-control prefill vs patched | disabled-control generation vs patched |
| ---: | ---: | ---: |
| 512 | -2.0% | +0.1% |
| 1024 | +5.3% | +0.2% |
| 2048 | +3.2% | +0.1% |
| 4096 | +3.4% | -0.5% |
| 8192 | +0.6% | -0.6% |

Conclusion: reject and do not port this Swival M5 barrier-elision patch. It
regresses the compact prefill median at most measured contexts, so a drift gate
is unnecessary.

## Q8_0 MPP Bug Triage: Block Size

Closed the first diagnostic from the older `m5-neural-accelerator` Phase 5
notes before revisiting any generic Q8_0 MPP kernel. The concern was that
Metal might pad:

```metal
struct block_q8_0 {
    half d;
    int8_t qs[32];
};
```

to something other than the host-side 34-byte row stride. A local runtime
Metal compile/run with `static_assert(sizeof(block_q8_0) == 34)` passed and
returned `34`.

Artifact:

- `speed-bench/local-runs/20260515-033017-q8-block-size-check/result.txt`

Conclusion: the old generic Q8_0 MPP bug is not explained by `block_q8_0`
padding. If that kernel is revisited, the next diagnostics should focus on
K-loop accumulation semantics and q8 dequant precision/layout, using the dense
Q8 comparator hook before any full-model timing.

## Q8_0 MPP Bug Triage: Static-K Accumulation

Ran a local runtime Metal harness for the next Phase 5 hypothesis: whether
`mpp::tensor_ops::matmul2d` accumulates into the same cooperative tensor across
a manual static-`TILEK` K-loop.

Artifact:

- `speed-bench/local-runs/20260515-033248-mpp-kloop-accum-check/result.txt`

The harness compares three half x half -> float kernels on the same
`M=64, N=32, K=128` tile:

- `k_full`: one dynamic-K `matmul2d` call;
- `k_loop`: four default-mode `TILEK=32` `matmul2d.run()` calls into the
  same zeroed cooperative tensor;
- `k_loop_mac`: the same static K-loop but with
  `matmul2d_descriptor::mode::multiply_accumulate`, matching this branch's
  existing Tensor kernels.

Result:

| Comparison | max abs | rms |
| --- | ---: | ---: |
| `kloop_vs_full` | 0.240234 | 0.101835 |
| `kloop_mac_vs_full` | 0 | 0 |
| `full_vs_host_f32` | 0 | 0 |
| `kloop_vs_host_f32` | 0.240234 | 0.101835 |
| `kloop_vs_host_last32` | 0 | 0 |
| `kloop_mac_vs_host_f32` | 0 | 0 |

Conclusion: default-mode static-`TILEK` `matmul2d.run()` calls overwrite with
the last K block rather than accumulating across the loop. The
`multiply_accumulate` descriptor mode accumulates correctly and matches both
dynamic-K `matmul2d` and the host fp32 reference for this shape. This branch's
existing Tensor kernels already use `multiply_accumulate`, so they are not
exposed to this specific failure. If the older generic Q8_0 MPP prototype is
revisited, verify it uses `multiply_accumulate` plus explicit cooperative-tensor
zeroing before moving on to dequant precision/layout diagnostics.

## Q8_0 MPP Bug Triage: Dequantized Tile Correctness

Ran a standalone q8_0 -> threadgroup-half -> `matmul2d` harness using the
corrected `multiply_accumulate` descriptor. The kernel uses the same q8_0 block
layout (`sizeof(block_q8_0) == 34`), dequantizes each 32-K weight block into a
`TN x TILEK` threadgroup half tile, then accumulates a `64 x 32 x 128` half x
half -> float matmul. The host reference mirrors DS4's legacy prefill math:
activations are half-rounded, q8 weights are dequantized in float and rounded
to half before fp32 accumulation.

Artifact:

- `speed-bench/local-runs/20260515-033841-q8-mpp-correctness-check/result.txt`

Result:

| Comparison | max abs | rms |
| --- | ---: | ---: |
| `q8_mpp_vs_host_half_reference` | 0 | 0 |

Conclusion: the corrected static-K q8_0 MPP tile is numerically sound in a
standalone harness. This does not promote a production Q8_0 Tensor route, but
it narrows the old failure down to implementation details rather than a
fundamental `block_q8_0` layout or `matmul2d` accumulation issue. The next
production experiment, if any, should be a default-off single instantiation of
the existing generic `kernel_mul_mm_mpp` for q8_0, gated through the dense Q8
comparator before any whole-model timing or drift gate.

## Rejected Q8_0 Generic MPP Matmul Route

Tried the proposed default-off single-instantiation generic Q8_0 MPP route
locally, then removed the production hook/template because timing was not
competitive with the current Tensor default.

Correctness/comparator artifacts:

- `speed-bench/local-runs/20260515-034306-manual-mpp-compare-probe/mpp-compare-summary.md`
- `speed-bench/local-runs/20260515-034322-manual-mpp-compare-probe/mpp-compare-summary.md`
- `speed-bench/local-runs/20260515-034336-manual-mpp-compare-probe/mpp-compare-summary.md`
- `speed-bench/local-runs/20260515-034411-manual-mpp-compare-probe/mpp-compare-summary.md`

The long `attn_q_b` probe compared all 43 layers with no breaches; worst max
abs was `3.57628e-06` and worst RMS was `7.3025e-08`. The long `attn_out`
probe also compared all 43 layers with no breaches; worst max abs was
`0.000335693` and worst RMS was `3.16847e-06`.

Timing artifacts:

- `speed-bench/local-runs/20260515-040005-experimental-q8-attn-q-b/prefill-candidate-summary.md`
- `speed-bench/local-runs/20260515-040427-experimental-q8-attn-out/prefill-candidate-summary.md`

Median speed vs current Tensor default:

| Candidate | 512 | 1024 | 2048 | 4096 | 8192 | Generation range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `attn_q_b` Q8_0 MPP | -8.4% | -5.8% | -1.6% | -0.7% | -0.0% | -0.4%..-0.1% |
| `attn_out` Q8_0 MPP | -6.2% | -7.6% | -3.7% | -1.0% | +0.3% | -0.8%..+0.4% |

Conclusion: reject before the five-fixture drift gate. The corrected MPP tile is
locally accurate, but the whole-kernel path regresses compact prefill where it
matters most and only reaches noise-level parity at 8192 tokens. Keeping a
default-off Q8_0 Tensor route would add surface area without a usable speed
tradeoff.

Post-cleanup validation:

- `make ds4 ds4-bench`
- `python3 -m py_compile speed-bench/*.py`
- `git diff --check`
- `python3 speed-bench/run_quality_drift_gate.py --max-tensor-standard-rms 0.30 --max-tensor-standard-top20-abs 0.60`

Fresh drift artifact:

- `speed-bench/local-runs/20260515-041151-quality-drift-gate/summary.md`
- `speed-bench/local-runs/20260515-041450-local-run-index/local-run-index.md`

Post-cleanup Tensor-vs-standard drift:

| Metric | Result |
| --- | ---: |
| top-1 mismatches | 0 |
| greedy mismatches | 0 |
| min top20 overlap | 19/20 |
| worst RMS | 0.239946 |
| worst top20 max abs | 0.55422 |

Gate result: OK.

## Rejected Legacy Routed-MoE Gate/Up Pair Kernel

Tried a default-off legacy `simdgroup_multiply_accumulate` pair kernel for the
early routed-MoE gate/up projections. The design preserved the reference
reduction shape for each projection while reusing the same activation tile for
gate and up. It was intended to target the early `0/0/0` window without taking
the drift-prone cooperative-Tensor route.

Comparator artifact:

- `speed-bench/local-runs/20260515-042045-manual-mpp-compare-probe/mpp-compare-summary.md`

The long `long_code_audit` comparator run covered `40` gate and `40` up
comparisons with no target breaches. Worst max abs was `8.39233e-05` and worst
RMS was `2.10939e-06`.

Timing artifact:

- `speed-bench/local-runs/20260515-042136-experimental-moe-legacy-pair-gate-up/prefill-candidate-summary.md`
- `speed-bench/local-runs/20260515-042900-local-run-index/local-run-index.md`

Median speed vs current Tensor default:

| 512 | 1024 | 2048 | 4096 | 8192 | Generation range |
| ---: | ---: | ---: | ---: | ---: | ---: |
| +0.5% | -4.5% | -4.6% | -0.4% | -0.9% | -2.1%..+0.4% |

Conclusion: reject before the five-fixture drift gate and remove the
experimental kernel/hook. The pair kernel was locally close to the reference,
but register pressure and the second accumulated output likely outweighed the
saved activation staging; it regressed the compact mid-contexts and generation
instead of improving prefill.

## Rechecked MoE Sum6 Boundary

Rechecked the existing `DS4_METAL_MOE_SUM6_DISABLE=1` control after the current
Tensor default changes, because the routed-MoE sum stage remains a possible
direct-down-sum target.

Artifact:

- `speed-bench/local-runs/20260515-043038-disable-moe-sum6-control/prefill-candidate-summary.md`

Median speed vs current Tensor default:

| 512 | 1024 | 2048 | 4096 | 8192 | Generation range |
| ---: | ---: | ---: | ---: | ---: | ---: |
| +0.9% | +5.5% | +4.0% | -0.3% | -0.7% | -1.0%..+0.1% |

This differs from the older boundary sweep enough to test a thresholded
candidate. A local patch added `DS4_METAL_MOE_SUM6_MIN_TOKENS=4096`, keeping
the fused `sum6` kernel for larger batches and using the generic add chain
below the threshold.

Threshold artifact:

- `speed-bench/local-runs/20260515-043605-moe-sum6-min4096/prefill-candidate-summary.md`
- `speed-bench/local-runs/20260515-044100-local-run-index/local-run-index.md`

Threshold result vs current Tensor default:

| 512 | 1024 | 2048 | 4096 | 8192 | Generation range |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -1.1% | -2.0% | +0.5% | +0.0% | -0.5% | -0.4%..+0.0% |

Conclusion: reject and remove the threshold knob before the five-fixture drift
gate. The all-disabled control shows the sum stage is noisy enough to revisit,
but the obvious token-threshold policy does not produce a clean compact prefill
win. A future direct-down-sum kernel still needs to beat the current fused
`sum6` baseline, not the slower generic fallback.

## Rejected Prefill Direct Down-Sum Probe

Tried a local default-off probe that reused the existing six-expert direct
down-sum kernel for batched prefill (`DS4_METAL_MOE_PREFILL_DIRECT_DOWN_SUM=1`)
instead of writing per-expert down outputs and running the separate `sum6`
kernel. The probe also forced the MoE mid buffer back to F32 because the
existing direct-sum kernels read F32 activations.

Short screen artifact:

- `speed-bench/local-runs/20260515-044921-moe-prefill-direct-down-sum/prefill-candidate-summary.md`

One-repeat screen vs current Tensor default:

| 512 | 1024 | 2048 | Generation range |
| ---: | ---: | ---: | ---: |
| -19.7% | -20.1% | -29.6% | -0.9%..+1.4% |

Conclusion: reject before the five-fixture drift gate and remove the temporary
hook. Saving the down scratch write plus sum dispatch does not compensate for
giving up the grouped prefill matmul; a production direct-down-sum design would
need to keep batched matmul throughput while accumulating directly into the
token-major output.

## Rejected Dense Q8_0 F16-RHS Prepack Probe

Tried a local default-off dense Q8_0 prefill probe that prepacked the RHS
activation matrix to half once, then ran a legacy simdgroup-MMA Q8_0 matmul
variant that read half RHS values. This preserved the same effective MMA input
precision as the current kernel, which casts F32 activations to half inside
each threadgroup, but added one F32-to-F16 prepack dispatch and a scratch RHS
buffer.

Short screen artifacts:

- `speed-bench/local-runs/20260515-045423-q8-f16-rhs-attn-q-b/prefill-candidate-summary.md`
- `speed-bench/local-runs/20260515-045455-q8-f16-rhs-attn-out/prefill-candidate-summary.md`

One-repeat screen vs current Tensor default:

| Candidate | 512 | 1024 | 2048 | Generation range |
| --- | ---: | ---: | ---: | ---: |
| `attn_q_b` F16 RHS | -3.2% | -0.0% | +0.2% | +0.0%..+0.7% |
| `attn_out` F16 RHS | -5.6% | -6.6% | -5.3% | -0.4%..+0.2% |

Conclusion: reject before the five-fixture drift gate and remove the temporary
kernel/hook. The prepack dispatch does not amortize at compact contexts, and
the only positive point is noise-level on `attn_q_b` at 2048 tokens.

## Rejected FlashAttention GPU Mask Fill

Tried a local default-off static-mixed FlashAttention mask-fill kernel
(`DS4_METAL_FLASH_ATTN_GPU_MASK_FILL=1`). The goal was to replace the CPU write
of the full transient half mask with a GPU analytic fill while leaving the
existing pad, block-map, and attention kernels unchanged.

Short screen artifact:

- `speed-bench/local-runs/20260515-045825-flash-attn-gpu-mask-fill/prefill-candidate-summary.md`

One-repeat screen vs current Tensor default:

| 512 | 1024 | 2048 | Generation range |
| ---: | ---: | ---: | ---: |
| -1.6% | -0.1% | -0.5% | -0.4%..+1.2% |

Conclusion: reject before the five-fixture drift gate and remove the temporary
kernel/hook. Moving mask fill to a separate GPU dispatch did not beat the CPU
fill path at compact contexts; the FlashAttention setup work still needs a more
integrated redesign if it is worth targeting.

## Rejected Routed-MoE Down-0 Window

Rechecked one remaining env-only routed-MoE window after the current Tensor
cleanup: move only the down projection to layer 0 while leaving gate/up on the
conservative default window (`DS4_METAL_MPP_MOE_DOWN_START_LAYER=0`). A short
screen looked plausible, so the candidate was run through the full two-repeat
candidate gate and five-fixture drift gate.

Artifacts:

- short screen:
  `speed-bench/local-runs/20260515-050301-moe-down0-gate15-up15-screen/prefill-candidate-summary.md`
- full gate:
  `speed-bench/local-runs/20260515-050334-moe-down0-gate15-up15/prefill-candidate-summary.md`

Median speed vs current Tensor default:

| 512 | 1024 | 2048 | 4096 | 8192 | Generation range |
| ---: | ---: | ---: | ---: | ---: | ---: |
| +5.6% | +6.0% | +0.0% | +2.0% | +1.2% | -2.6%..-0.0% |

Promotion decision: reject. The repeat-level speed floor failed at 2048 and
8192 (`min repeat=-4.0%`), and the five-fixture drift gate failed:
`long_memory_archive` changed top-1 and greedy step 0, Tensor-vs-standard worst
RMS rose to `0.550345`, and worst top20 abs rose to `1.38147`. This confirms
that simply extending the current Tensor down route into the early layers is
not a production path; early routed-MoE needs a reference-compatible kernel
design, not another window expansion.

An adjacent short screen with `DS4_METAL_MPP_MOE_DOWN_START_LAYER=4` also
failed before drift:

- `speed-bench/local-runs/20260515-051113-moe-down4-gate15-up15-screen/prefill-candidate-summary.md`

That run was +3.5% at 512 and +3.2% at 1024, but -0.3% at 2048 with a -5.3%
generation point. Excluding layers 0..3 therefore does not recover a clean
early-down production candidate.

The drift-mitigation variant
`DS4_METAL_MPP_MOE_DOWN_START_LAYER=0 DS4_METAL_MOE_MID_F32=1` also failed the
short speed screen before drift:

- `speed-bench/local-runs/20260515-051250-moe-down0-mid-f32-screen/prefill-candidate-summary.md`

It measured +4.1% at 512 and +3.3% at 1024, but -0.4% at 2048. Preserving the
F32 routed intermediate is therefore not a usable way to make the down-0 window
production-safe.

## Rejected Mul-MM-ID Writeback Index Probe

Tried a local default-off function-constant probe that changed the generic
`kernel_mul_mm_id` writeback column assignment from `sgitg` to `tiitg/32`,
matching the separate fast-layout kernel's writeback loop while preserving the
same matmul arithmetic and result layout.

Short screen artifact:

- `speed-bench/local-runs/20260515-051517-mul-mm-id-writeback-tiidx-screen/prefill-candidate-summary.md`

One-repeat screen vs current Tensor default:

| 512 | 1024 | 2048 | Generation range |
| ---: | ---: | ---: | ---: |
| -5.6% | +0.1% | -0.5% | -0.4%..+3.7% |

Conclusion: reject before drift and remove the temporary hook. This writeback
mapping is arithmetic-neutral but not a prefill win; the generic routed-MoE
kernel still needs a real staging or output-layout change rather than a
thread-index assignment tweak.

## Rejected Legacy Gate/Up Pair Probe

Tried a local default-off `DS4_METAL_MOE_PAIR_GATE_UP_LEGACY=1` probe that
computed routed-MoE gate and up in one legacy simdgroup-MMA kernel for early
non-MPP layers. The goal was to preserve the standard Metal reduction order
while reusing the shared expert map and activation tile.

Comparator spot checks on `long_memory_archive` matched the existing legacy
matmuls for the first large layer-0 projections:

- `moe_gate`: `max_abs=0`, `rms=0`;
- `moe_up`: `max_abs=0`, `rms=0`.

Speed-screen artifact:

- `speed-bench/local-runs/20260515-072058-moe-pair-gate-up-legacy-v2/prefill-candidate-summary.md`

Two-repeat compact screen vs current Tensor default:

| 512 | 1024 | 2048 | 4096 | 8192 | Generation range |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -0.9% | +0.2% | +1.5% | +2.5% | +1.9% | -1.2%..+0.3% |

Repeat-level prefill still dipped negative at every measured context except
the 512-token median was already negative: min repeat was `-1.3%`. Conclusion:
reject before the five-fixture drift gate and remove the temporary kernel. The
pairing idea is locally equivalent but not repeat-stable enough to carry as a
default-off production candidate.

## Current Default Chart Refresh, Timestamped Local Artifact

Regenerated the current branch standard/quality/Tensor chart with the updated
`speed-bench/run_metal_tensor_bench.sh` defaults. The script now writes
timestamped artifacts under ignored `speed-bench/local-runs/` instead of
`/tmp`, so multiple comparison runs can be kept locally without pushing them.

Command:

```sh
OPEN_CHART=0 speed-bench/run_metal_tensor_bench.sh
```

Artifact root:

- `speed-bench/local-runs/20260515-052156-metal-tensor-bench/`

Chart:

- `speed-bench/local-runs/20260515-052156-metal-tensor-bench/20260515-052156_gen128_ds4_bench_standard_quality_tensor.png`

Tensor default remains a broad prefill win over standard Metal with only a
small generation tax:

| ctx | Tensor prefill vs standard | Tensor generation vs standard |
| ---: | ---: | ---: |
| 512 | +30.2% | -0.5% |
| 1024 | +31.4% | -1.3% |
| 2048 | +26.3% | -1.0% |
| 4096 | +22.1% | -0.9% |
| 8192 | +20.1% | -0.7% |
| 16384 | +19.4% | -0.8% |
| 32768 | +17.7% | -0.6% |
| 65536 | +15.1% | -0.6% |

## Compact Current Stage Profile

Reran the current Tensor default stage profile on `long_code_audit` at
`-c 8192` after the earlier oversized-prompt attempt failed. This uses the
same 3844-token prompt as the 16k profile while keeping the context closer to
the middle of the benchmark sweep.

Command:

```sh
env DS4_METAL_MOE_STAGE_PROFILE=1 \
    DS4_METAL_Q8_PREFILL_PROFILE=1 \
    DS4_METAL_Q8_PREFILL_PROFILE_FILTER=attn_ \
    DS4_METAL_FLASH_ATTN_STAGE_PROFILE=1 \
    DS4_METAL_LAYER_PROFILE=1 \
    ./ds4 --metal -mt auto \
      --prompt-file tests/test-vectors/prompts/long_code_audit.txt \
      -c 8192 -n 1 --system "" --nothink --temp 0
```

Artifacts:

- `speed-bench/local-runs/20260515-053713-current-ctx8192-stage-profile/run.log`
- `speed-bench/local-runs/20260515-053713-current-ctx8192-stage-profile/stage-profile-summary.md`
- `speed-bench/local-runs/20260515-053713-current-ctx8192-stage-profile/stage-profile-summary.json`

Result: `420.33` prefill t/s, `603` parsed profile events, and
`5011.795 ms` parsed stage time. The compact profile matches the earlier 16k
profile: routed-MoE gate/up/down and the two large dense Q8_0 attention
projections remain the dominant prefill cost.

| Stage | total ms | events | avg ms |
| --- | ---: | ---: | ---: |
| `moe_stage.gate` | 909.794 | 43 | 21.158 |
| `moe_stage.up` | 909.728 | 43 | 21.156 |
| `moe_stage.down` | 834.073 | 43 | 19.397 |
| `q8.attn_out` | 803.923 | 43 | 18.696 |
| `q8.attn_q_b` | 797.692 | 43 | 18.551 |
| `flash_attn.static_mixed_nonvec.attention` | 310.597 | 20 | 15.530 |

MoE timing by Tensor mask:

| MoE mpp mask | top stages | total ms |
| --- | --- | ---: |
| `0/0/0` | `up`=412.5, `gate`=409.3, `down`=409.1 | 1268.948 |
| `1/1/1` | `gate`=400.4, `up`=397.5, `down`=383.9 | 1256.632 |
| `0/0/1` | `gate`=100.0, `up`=99.7, `down`=41.0 | 248.767 |

Conclusion: the next production candidate should not be another route-window
or tile-size sweep. Those have been exhausted and either fail speed stability
or the five-fixture drift gate. The remaining plausible prefill work is a
reference-compatible routed-MoE or dense Q8_0 kernel redesign that keeps the
current low-drift arithmetic envelope while reducing staging/writeback cost.

## Bench-Prompt Current Stage Profile

Reran the stage profiler on the same `speed-bench/promessi_sposi.txt` prompt
used by the chart and candidate gate, walking the 512..8192 frontiers in one
Tensor run. This checks that the hotspot ranking from the smaller fixture also
holds on the actual speed-gate workload.

Command:

```sh
env DS4_METAL_MOE_STAGE_PROFILE=1 \
    DS4_METAL_Q8_PREFILL_PROFILE=1 \
    DS4_METAL_Q8_PREFILL_PROFILE_FILTER=attn_ \
    DS4_METAL_FLASH_ATTN_STAGE_PROFILE=1 \
    DS4_METAL_LAYER_PROFILE=1 \
    ./ds4-bench -mt auto \
      --prompt-file speed-bench/promessi_sposi.txt \
      --ctx-start 512 --ctx-max 8192 --gen-tokens 1
```

Artifacts:

- `speed-bench/local-runs/20260515-073001-current-promessi-stage-profile/bench.csv`
- `speed-bench/local-runs/20260515-073001-current-promessi-stage-profile/stage-profile-summary.md`
- `speed-bench/local-runs/20260515-073001-current-promessi-stage-profile/stage-profile-summary.json`

Parsed profile result: `3071` events and `11745.870 ms` parsed stage time.
The profile confirms the same target order as the previous current-default
profile:

| Stage | total ms | share |
| --- | ---: | ---: |
| `moe_stage.up` | 2519.278 | 21.4% |
| `moe_stage.gate` | 2511.646 | 21.4% |
| `moe_stage.down` | 2279.191 | 19.4% |
| `q8.attn_out` | 1790.328 | 15.2% |
| `q8.attn_q_b` | 1723.122 | 14.7% |
| `flash_attn.static_mixed_nonvec.attention` | 77.665 | 0.7% |

MoE by Tensor mask:

| MoE mpp mask | top stages | total ms |
| --- | --- | ---: |
| `0/0/0` | `up`=1151.6, `gate`=1146.8, `down`=1120.8 | 3521.858 |
| `1/1/1` | `up`=1090.0, `gate`=1086.5, `down`=1049.6 | 3454.142 |
| `0/0/1` | `gate`=278.4, `up`=277.7, `down`=108.7 | 689.084 |

Decision: keep FlashAttention work deprioritized for prefill on this branch.
The next production candidate still needs to attack routed-MoE or dense Q8_0
matmul. Within routed-MoE, the early `0/0/0` window remains the best target,
but the rejected legacy gate/up pair shows that simply combining two reference
matmuls is not enough; the next kernel must reduce staging/writeback cost
without changing the low-drift arithmetic envelope.

## Continuation-Chunk Routed-MoE Probe

Tried a position-filtered routed-MoE policy that keeps the current conservative
default window at `pos=0`, but uses the fast all-layer routed-MoE profile on
later prefill chunks:

```sh
DS4_METAL_MPP_FAST=1
DS4_METAL_MPP_MOE_GATE_FILTER=layer=15-42,pos=512,pos=1024,pos=2048,pos=4096
DS4_METAL_MPP_MOE_UP_FILTER=layer=15-42,pos=512,pos=1024,pos=2048,pos=4096
DS4_METAL_MPP_MOE_DOWN_FILTER=layer=12-42,pos=512,pos=1024,pos=2048,pos=4096
```

Artifact:

- `speed-bench/local-runs/20260515-073209-mpp-fast-continuation-chunks/prefill-candidate-summary.md`

Two-repeat compact screen vs current Tensor default:

| 512 | 1024 | 2048 | 4096 | 8192 | Generation range |
| ---: | ---: | ---: | ---: | ---: | ---: |
| +4.2% | +24.0% | +13.3% | +13.6% | +8.3% | -0.7%..+0.8% |

Repeat-level prefill was positive at every measured point; min repeat prefill
was `+1.5%`. The usual five-fixture drift gate also stayed green with the same
Tensor-vs-standard summary as the current default: top1 mismatches `0`, greedy
mismatches `0`, worst RMS `0.239946`, and worst top20 abs `0.55422`.

Important caveat: this is not production-safe on the current evidence. The
five fixtures mostly exercise `pos=0`, while this candidate's new behavior is
the nonzero-position continuation chunks. `run_prefill_candidate_gate.py` now
marks nonzero `pos=` candidates as not promotion-safe until a chunked or
long-prompt drift check covers that route. Keep this as a promising
default-off direction, not an auto-policy change.

## Dense Q8_0 Comparator Hook Refresh

The earlier dense Q8_0 comparator notes were stale relative to the current
code: the README documented `DS4_METAL_Q8_COMPARE=1`, but the active Q8 path
only had profiling (`DS4_METAL_Q8_PREFILL_PROFILE=1`). Restored the default-off
compare hook in `ds4_gpu_matmul_q8_0_tensor()` and wired
`run_mpp_compare_probe.py --route q8 --q8-filter <substring>` so future dense
Q8_0 kernel attempts can be checked locally before the five-fixture drift gate.

Smoke command:

```sh
python3 speed-bench/run_mpp_compare_probe.py \
  --case short_code_completion \
  --route q8 \
  --q8-filter attn_q_b \
  --compare-max 3 \
  --verbose \
  --top 10
```

Artifact:

- `speed-bench/local-runs/20260515-054611-manual-mpp-compare-probe/mpp-compare-summary.md`

Result: `3` parsed `q8` comparisons for `attn_q_b`, no target breaches,
and zero delta against the current legacy candidate/reference path:

| Route | Module | Shape | Max abs | RMS |
| --- | --- | --- | ---: | ---: |
| `q8` | `layer=0 pos=0 attn_q_b` | `32768x27x1024` | 0 | 0 |
| `q8` | `layer=1 pos=0 attn_q_b` | `32768x27x1024` | 0 | 0 |
| `q8` | `layer=2 pos=0 attn_q_b` | `32768x27x1024` | 0 | 0 |

## Rejected Dense Q8_0 Tok64 MPP Probe

Tried a local default-off Q8_0 Metal Tensor tile that swapped the previous
generic MPP shape from `64x32` output-row/token tiles to `32x64`, aiming to
reuse q8 dequantized rows across a wider token tile. The temporary hook used:

```sh
DS4_METAL_Q8_MPP_TOK64=1
DS4_METAL_Q8_MPP_TOK64_FILTER=<attn_q_b|attn_out>
```

Comparator smoke artifacts:

- `speed-bench/local-runs/20260515-055108-manual-mpp-compare-probe/mpp-compare-summary.md`
- `speed-bench/local-runs/20260515-055201-manual-mpp-compare-probe/mpp-compare-summary.md`

The local comparator was clean before timing. For `attn_q_b`, the first three
layers had worst max abs `1.13249e-06` and worst RMS `2.32904e-08`. For
`attn_out`, the first three layers had worst max abs `2.95639e-05` and worst
RMS `2.98521e-06`.

Short timing artifacts:

- `speed-bench/local-runs/20260515-055126-q8-mpp-tok64-attn-q-b-screen/prefill-candidate-summary.md`
- `speed-bench/local-runs/20260515-055212-q8-mpp-tok64-attn-out-screen/prefill-candidate-summary.md`

One-repeat timing versus the current Tensor default:

| Candidate | 512 | 1024 | 2048 | Generation range |
| --- | ---: | ---: | ---: | ---: |
| `attn_q_b` tok64 MPP | -5.1% | +0.2% | +0.0% | -0.7%..-0.1% |
| `attn_out` tok64 MPP | -5.9% | -8.1% | -5.8% | -0.1%..+2.7% |

Conclusion: reject before the five-fixture drift gate and remove the temporary
kernel/hook. The wider token tile was locally accurate, but it did not improve
compact prefill; `attn_q_b` only reached noise-level parity after a short-context
regression, and `attn_out` regressed all measured compact contexts.

## Rejected Dense Q8_0 64x64 MPP Probe

Tried the other plausible MPP tile shape in the same family: `64x64`
output-row/token tiles. This kept the output-row width of the earlier generic
MPP route while doubling token width, with a temporary default-off hook:

```sh
DS4_METAL_Q8_MPP_64X64=1
DS4_METAL_Q8_MPP_64X64_FILTER=<attn_q_b|attn_out>
```

Comparator smoke artifacts:

- `speed-bench/local-runs/20260515-055459-manual-mpp-compare-probe/mpp-compare-summary.md`
- `speed-bench/local-runs/20260515-055719-manual-mpp-compare-probe/mpp-compare-summary.md`

The first three `attn_q_b` layers were clean with worst max abs
`1.13249e-06` and RMS `2.32904e-08`. The first three `attn_out` layers were
also clean with worst max abs `2.95639e-05` and RMS `2.98521e-06`.

Timing artifacts:

- `speed-bench/local-runs/20260515-055512-q8-mpp-64x64-attn-q-b-screen/prefill-candidate-summary.md`
- `speed-bench/local-runs/20260515-055548-q8-mpp-64x64-attn-q-b-long-screen/prefill-candidate-summary.md`
- `speed-bench/local-runs/20260515-055730-q8-mpp-64x64-attn-out-screen/prefill-candidate-summary.md`

One-repeat timing versus the current Tensor default:

| Candidate | 512 | 1024 | 2048 | 4096 | 8192 | Generation range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `attn_q_b` 64x64 short | -4.0% | +0.7% | +0.3% | n/a | n/a | +0.4%..+4.0% |
| `attn_q_b` 64x64 long | +5.9% | +7.0% | -3.5% | -1.2% | +0.7% | -6.2%..+0.5% |
| `attn_out` 64x64 short | -1.6% | -0.3% | -1.0% | n/a | n/a | +0.5%..+0.8% |

Conclusion: reject before the five-fixture drift gate and remove the temporary
kernel/hook. The candidate was locally accurate, but not speed-stable: it
regressed compact `attn_out`, regressed `attn_q_b` at 512 in the short screen,
and the longer `attn_q_b` screen showed mid-context prefill regressions plus
generation-floor breaches.

## Rejected FlashAttention Fast CPU Mask Fill

Tried a local CPU-side prefill mask fill rewrite behind
`DS4_METAL_FLASH_ATTN_FAST_CPU_MASK_FILL=1`. The patch kept the same mask
values but replaced per-element causal/window branches with row fill plus
contiguous zero spans for visible raw and compressed keys.

Short timing artifact:

- `speed-bench/local-runs/20260515-060204-flash-attn-fast-cpu-mask-fill-screen/prefill-candidate-summary.md`

One-repeat timing versus the current Tensor default:

| 512 | 1024 | 2048 | Generation range |
| ---: | ---: | ---: | ---: |
| -0.6% | -0.1% | -0.2% | -0.3%..+0.0% |

Conclusion: reject before drift and remove the temporary hook. The rewrite was
math-identical, but the existing branchy fill is already efficient enough at
compact contexts; the row-fill/memset variant added overhead instead of saving
prefill time.

## Rejected M5 Private Scratch Buffers

Ported the `swival-ds4-m5/m5` private scratch-buffer idea as a local opt-in
candidate (`DS4_METAL_PRIVATE_SCRATCH=1`), keeping CPU-written masks and
attention-output group-id tables in shared storage. The change only affected
GPU-only scratch allocation storage mode, so arithmetic and drift risk were low,
but timing was not favorable.

Short timing artifact:

- `speed-bench/local-runs/20260515-060603-private-scratch-screen/prefill-candidate-summary.md`

One-repeat timing versus the current Tensor default:

| 512 | 1024 | 2048 | Generation range |
| ---: | ---: | ---: | ---: |
| -0.2% | -0.1% | -2.0% | -5.2%..-0.5% |

Conclusion: reject before the five-fixture drift gate and remove the temporary
hook. Private scratch storage did not improve compact prefill and introduced a
generation-floor miss at 1024 tokens.

## Rejected MoE Clamped-Activation Writeback

Screened the existing diagnostic `DS4_METAL_MOE_WRITE_CLAMPED_ACT=1` switch
after the compact stage profile showed `moe_stage.activation_weight` around one
percent of parsed prefill time. The normal release path already avoids writing
the clamped gate/up intermediates because no later inference stage consumes
them; this switch restores those writes only for intermediate-tensor
diagnostics.

Short timing artifact:

- `speed-bench/local-runs/20260515-061018-moe-write-clamped-act-screen/prefill-candidate-summary.md`

One-repeat timing versus the current Tensor default:

| 512 | 1024 | 2048 | Generation range |
| ---: | ---: | ---: | ---: |
| -0.1% | -0.5% | -0.5% | -1.1%..+0.8% |

Conclusion: reject before the five-fixture drift gate. The switch is useful for
diagnostics, but it is not a production optimization and confirms that the
default no-writeback activation path is already the right choice.

## Current Default Drift Gate Refresh

Reran the five-fixture quality drift gate after the local comparator/script
changes and the rejected activation-writeback screen. No rejected speed probe
was enabled for this run.

Command:

```sh
python3 speed-bench/run_quality_drift_gate.py \
  --no-fail \
  --out-dir speed-bench/local-runs/20260515-061111-current-default-quality-drift-gate
```

Artifacts:

- `speed-bench/local-runs/20260515-061111-current-default-quality-drift-gate/summary.md`
- `speed-bench/local-runs/20260515-061111-current-default-quality-drift-gate/summary.json`

Gate result: `OK`.

| Pair | Top1 mismatches | Greedy mismatches | Min top20 | Worst RMS | Worst top20 abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `standard_vs_quality` | 0 | 1 | 18/20 | 0.618172 | 2.24006 |
| `tensor_vs_quality` | 0 | 1 | 18/20 | 0.618172 | 2.24006 |
| `tensor_vs_standard` | 0 | 0 | 19/20 | 0.239946 | 0.55422 |

Conclusion: current default Tensor remains inside the strict Tensor-vs-standard
envelope (`0.30` RMS, `0.60` top20 abs) after the recent non-production
diagnostic and bench-script changes.

## Remaining Prefill-Audit Notes

Re-audited the current code and env surface after the rejected activation
writeback screen to avoid repeating low-value probes.

Dense Q8_0:

- The active prefill path is still `kernel_mul_mm_q8_0_f32`, a hand-written
  simdgroup-MMA kernel with a hard-coded `64x32` output-row/token tile.
- The four simdgroups are mapped over two 32-row halves and two 16-token halves,
  so changing the output-row tile is not a host-only knob; it requires a new
  simdgroup layout and a new kernel family.
- Already rejected Q8_0 scheduling/prototype axes include split-tail, token-64
  widening, generic MPP, direct-RHS Tensor, F16 RHS prepack, tok64 MPP, and
  `64x64` MPP.

FlashAttention:

- Static-mixed non-vector attention remains a secondary hotspot, but the
  low-risk setup/geometry probes have already been rejected: mask cache, CPU
  block map, NSG4, real `C=32`, real `Q=16`, GPU mask fill, and fast CPU mask
  fill.
- The remaining work is inside the attention kernel body, not another
  mask/setup toggle.

Env surface:

- `DS4_METAL_DISABLE_ROUTER_SELECT_FUSION` is decode-only for this branch's
  router fast path (`n_tokens == 1`), so it is not a prefill gate candidate.
- Startup/residency/hot-pipeline switches still affect warmup behavior rather
  than steady-state prefill throughput.

Conclusion: there is no obvious untested env-only or one-line prefill candidate
left. The next optimization pass should start as a new default-off kernel
family, with the dense Q8_0 comparator and the five-fixture drift gate as the
first acceptance checks.

## Rejected Dense Q8_0 Row-Pair Probe

Tried a local default-off dense Q8_0 kernel family that computed two adjacent
`64x32` output-row/token tiles in one threadgroup and shared the staged RHS tile
between them. The goal was to reduce RHS staging and dispatch overhead while
keeping each `64x32` tile's dequantization and simdgroup-MMA accumulation order
aligned with `kernel_mul_mm_q8_0_f32`.

Temporary hook:

```sh
DS4_METAL_Q8_ROWPAIR=1
DS4_METAL_Q8_ROWPAIR_FILTER=<attn_q_b|attn_out>
```

Comparator smoke artifacts:

- `speed-bench/local-runs/20260515-062046-manual-mpp-compare-probe/mpp-compare-summary.md`
- `speed-bench/local-runs/20260515-062103-manual-mpp-compare-probe/mpp-compare-summary.md`

The first three `attn_q_b` and `attn_out` layers were exact against the legacy
Q8_0 path: worst max abs `0`, RMS `0`.

Short timing artifacts:

- `speed-bench/local-runs/20260515-062116-q8-rowpair-attn-q-b-screen/prefill-candidate-summary.md`
- `speed-bench/local-runs/20260515-062148-q8-rowpair-attn-out-screen/prefill-candidate-summary.md`

One-repeat timing versus the current Tensor default:

| Candidate | 512 | 1024 | 2048 | Generation range |
| --- | ---: | ---: | ---: | ---: |
| `attn_q_b` row-pair | +0.3% | -0.8% | -4.1% | -2.4%..-0.5% |
| `attn_out` row-pair | -5.7% | -7.1% | -6.5% | -1.3%..-0.2% |

Conclusion: reject before the five-fixture drift gate and remove the temporary
kernel/hook. Sharing the RHS tile did not compensate for the extra accumulator
pressure and larger threadgroup footprint; it made `attn_out` consistently
slower and only gave a noise-level 512-token point on `attn_q_b`.

## Small-Batch Dense Boundary Audit

Checked the dense `mul_mv_ext` path before starting another prefill candidate.
Both Q8_0 and F16 Tensor dense wrappers route through `mul_mv_ext` only when
`n_tok <= 8` and the input dimension is divisible by 128. The compact prefill
gate starts at 512 tokens, and the Q8_0 profiling/comparator hooks are
deliberately scoped to `n_tok > 8`, so this helper is outside the measured
steady-state prefill route.

The F16 pair Tensor path also rejects `n_tok <= 8` for its batched pair-MPP
candidate and falls back to the single-output dense helper. The previously
audited FlashAttention vector helper has the same shape issue in the opposite
direction: it is kept below 20 tokens because forcing it into normal prefill
would allocate multi-GiB temporary buffers.

Conclusion: do not run a compact prefill timing gate for the small-batch dense
boundary. It may matter for prompt tails, speculative/MTP-style microbatches, or
decode-adjacent work, but it is not a promotion candidate for the current
512-token-and-up prefill benchmark.

## FlashAttention Static-Mixed Kernel Triage

Inspected the static-mixed non-vector prefill path after the routed-MoE and
dense Q8_0 frontier checks. The current path materializes a half mask on the
CPU, optionally copies a compressed mask into it, scans that mask with
`kernel_flash_attn_ext_blk`, then runs the generic
`kernel_flash_attn_ext_f16_dk512_dv512` non-vector attention kernel with
`has_mask=true`, `has_sinks=true`, `has_bias=false`, `has_scap=false`,
`nqptg=8`, `ncpsg=64`, and `nsg=8` for the DS4 512-wide heads.

Previously rejected FlashAttention probes already cover the simple knobs:

- `NCPSG=128`, real `C=32`, real `Q=16`, and `NSG=4` did not produce a compact
  whole-model prefill win;
- CPU/GPU mask-fill rewrites, mask caching, and CPU block-map generation either
  regressed speed or were noise-level;
- forcing the vector helper into normal prefill is not viable because its
  temporary buffer scales to multi-GiB at ordinary prefill sizes.

The remaining plausible attention target is therefore not another host toggle.
It is a new static-mixed-specific non-vector kernel that computes the raw
causal/window visibility and compressed-row visibility from `(q, k, ratio,
window)` inside the kernel, avoiding the materialized mask and block-map path
for the common unmasked static-mixed prefill case. This should be default-off
at first and must compare against the existing generic masked path before any
whole-model timing. Because it changes masking implementation rather than the
intended math, acceptance should require:

- local head-output comparator against the existing generic FlashAttention path
  on static-mixed fixtures;
- compact prefill timing versus current Tensor default;
- the five-fixture drift gate before promotion.

Conclusion: do not start another small FlashAttention flag screen. The next
attention optimization should be a separate static-mixed kernel family with
explicit local output comparison and the usual five-scenario drift gate.

## FlashAttention Comparator Hook

Added the local output comparator needed before implementing the
static-mixed-specific attention kernel family. The hook is default-off and does
not change normal inference:

```sh
DS4_METAL_FLASH_ATTN_COMPARE=1
DS4_METAL_MPP_COMPARE_ROUTE=flash_attn
DS4_METAL_FLASH_ATTN_COMPARE_FILTER=<optional substring>
```

When enabled, the current candidate head output is snapshotted and the existing
generic static-mixed FlashAttention path is replayed into a reference buffer on
the same command buffer. The result is registered through the same comparator
summary path used by routed-MoE, attention-output, and dense Q8_0 probes. The
graph now sets compare context around the static-mixed prefill attention call,
so reports include the layer and `pos0` context.

`speed-bench/run_mpp_compare_probe.py` also accepts `--route flash_attn` and
`--flash-attn-filter ...`, which enables the hook and writes the usual
`mpp-compare-summary.md/json` artifacts under `speed-bench/local-runs/`.

Smoke command:

```sh
python3 speed-bench/run_mpp_compare_probe.py \
  --case short_code_completion \
  --route flash_attn \
  --flash-attn-filter static_mixed \
  --compare-max 1 \
  --gen-tokens 1 \
  --verbose
```

Artifact:

- `speed-bench/local-runs/20260515-063525-manual-mpp-compare-probe/mpp-compare-summary.md`

Result: one `flash_attn` comparison on layer 2, shape `512x64x27`, with max abs
`0`, RMS `0`, and no nonfinite values.

This is scaffolding only: the current default still runs the generic
static-mixed path. No speed or drift gate was run for this change because it is
inactive unless the diagnostic env is set.

## Rejected FlashAttention Analytic Static Mask Probe

Tried a default-off analytic static-mixed mask path that skipped the
materialized mask and block-map for unmasked static-mixed prefill. Local
comparator checks first exposed a mixed raw/compressed boundary bug, then passed
after forcing the crossing block through per-element masking:

- `speed-bench/local-runs/20260515-064033-manual-mpp-compare-probe/mpp-compare-summary.md`
- `speed-bench/local-runs/20260515-064229-manual-mpp-compare-probe/mpp-compare-summary.md`

The short speed screen failed before the drift gate:

- `speed-bench/local-runs/20260515-064253-flash-attn-static-mask-screen/prefill-candidate-summary.md`

One-repeat timing versus the current Tensor default:

| Context | Prefill delta | Generation delta |
| --- | ---: | ---: |
| 512 | -11.9% | +1.0% |
| 1024 | -5.5% | +0.2% |
| 2048 | -5.1% | +2.3% |

Conclusion: reject and remove the production hook. The local comparator
scaffold remains useful, but this analytic-mask variant is slower on the
prefill target, so no five-fixture drift gate was run.

## Post-Cleanup Frontier Check

Re-smoked the FlashAttention comparator after removing the rejected analytic
static-mask hook:

```sh
python3 speed-bench/run_mpp_compare_probe.py \
  --case short_code_completion \
  --route flash_attn \
  --flash-attn-filter static_mixed \
  --compare-max 1 \
  --gen-tokens 1 \
  --verbose
```

Artifact:

- `speed-bench/local-runs/20260515-065041-manual-mpp-compare-probe/mpp-compare-summary.md`

Result: one static-mixed prefill comparison on layer 2, shape `512x64x27`,
max abs `0`, RMS `0`, no nonfinite values. The comparator scaffold is still
valid for future FlashAttention kernel work.

Also wrote a timestamped local-run index:

- `speed-bench/local-runs/20260515-065056-local-run-index/local-run-index.md`
- `speed-bench/local-runs/20260515-065625-local-run-index/local-run-index.md`

The candidate gate now enforces the speed-first workflow before nested drift
runs. Verification used the saved rejected `f16-pair-current` run with
`--reuse --run-drift-gate --no-fail`; it reused existing CSVs, did not run the
model, skipped the drift gate, and wrote the skip reason into the ignored local
summary:

- `speed-bench/local-runs/20260514-171939-f16-pair-current/prefill-candidate-summary.md`

The Markdown scorecard repeat table was validated by regenerating the saved
`mpp-gateup0-3-down12` candidate with `--reuse`. The report now shows the exact
repeat-level cause for skipping drift: at 512 tokens, repeat prefill deltas were
`-0.5%` and `+3.9%` even though the median was `+1.7%`.

- `speed-bench/local-runs/20260515-065835-mpp-gateup0-3-down12/prefill-candidate-summary.md`

The local-run index now mirrors that stricter screen by showing both median and
repeat-level minimum prefill deltas. This keeps median-positive but
repeat-unstable candidates visible as rejected in the top-level artifact index,
instead of requiring a separate JSON lookup.

- `speed-bench/local-runs/20260515-070910-local-run-index/local-run-index.md`

Important caveat from that index: older host-only FlashAttention tile screens,
such as `flash-attn-ncpsg32`, can still appear near the top by speed. Do not
revive those directly. The later real specializations with matching host and
Metal template geometry were tested in `Rejected FlashAttention Tile Variants`
and did not meet the compact prefill speed bar.

Current frontier remains the early routed-MoE `0/0/0` window. The existing MPP
fast-layout gate/up/down route is fast but fails the strict Tensor-vs-standard
drift envelope when expanded into early layers. A useful next kernel must
therefore preserve the standard simdgroup-MMA arithmetic closely while reducing
the early-window gate/up/down cost; another route-window scan or stale
FlashAttention geometry flag is unlikely to be productive.

## Continuation-Chunk Drift Gate

Added a resumed-prefill drift gate for candidates that only route nonzero
`pos=` chunks:

```sh
python3 speed-bench/run_chunked_prefill_drift_gate.py \
  --preset mpp-fast-continuation-chunks \
  --max-tensor-standard-rms 0.30 \
  --max-tensor-standard-top20-abs 0.60 \
  --no-fail
```

Artifacts:

- `speed-bench/local-runs/20260515-074852-mpp-fast-continuation-chunks-chunked-drift-gate/summary.md`
- `speed-bench/local-runs/20260515-075200-local-run-index/local-run-index.md`

The candidate still has no top-1 mismatch at resumed frontiers, but it fails
the strict Tensor-vs-standard drift envelope:

| Frontier | Same top1 | Top20 | RMS | Top20 abs |
| ---: | --- | ---: | ---: | ---: |
| 512 | yes | 19/20 | 0.202659 | 0.579939 |
| 1024 | yes | 19/20 | 0.707456 | 1.95875 |
| 2048 | yes | 18/20 | 0.451973 | 1.25351 |
| 4096 | yes | 18/20 | 0.382888 | 1.08998 |
| 8192 | yes | 19/20 | 0.409673 | 0.654034 |

Conclusion: reject `mpp-fast-continuation-chunks` for production promotion.
The speed gain is real, but the newly covered resumed chunks drift too far from
standard Metal. Keep the new gate for future nonzero-`pos` candidates.

Follow-up tooling change: `run_prefill_candidate_gate.py --run-drift-gate` now
detects nonzero `pos=` route filters and runs this chunked frontier gate after
the speed screen passes. The promotion scorecard treats missing or failing
chunked coverage as a blocker for that class of candidate, so future
continuation-prefill experiments cannot pass on the five-fixture gate alone.

Regenerated the original `mpp-fast-continuation-chunks` candidate scorecard
with the integrated nested chunked gate:

- `speed-bench/local-runs/20260515-073209-mpp-fast-continuation-chunks/prefill-candidate-summary.md`
- `speed-bench/local-runs/20260515-073209-mpp-fast-continuation-chunks/chunked-drift-gate/summary.md`
- `speed-bench/local-runs/20260515-081337-local-run-index/local-run-index.md`
- `speed-bench/local-runs/20260515-081533-local-run-index/local-run-index.md`

The promotion decision now reports the actual blocker directly: the candidate
passes the speed screen and the five-fixture drift gate, but fails chunked
Tensor-vs-standard drift at frontier `1024` with worst RMS `0.707456` and worst
top20 abs `1.95875`. The local-run index now separates five-fixture drift from
coverage drift, so this candidate appears as `5-fixture OK=yes` but
`Coverage OK=no` instead of looking drift-clean in the speed table.

Follow-up baseline check: the current default Tensor path itself does not meet
the strict absolute chunked Tensor-vs-standard envelope on resumed frontiers,
so coverage for candidate env overrides now uses candidate Tensor versus the
current no-env Tensor baseline instead of candidate Tensor versus standard
Metal. The standalone chunked gate still reports all pairs, but when env
overrides are present it also captures `default_tensor` and reports
`tensor_vs_default_tensor`.

Artifacts:

- `speed-bench/local-runs/20260515-081710-current-default-chunked-drift-gate/summary.md`
- `speed-bench/local-runs/20260515-073209-mpp-fast-continuation-chunks/chunked-drift-gate/summary.md`

Current default chunked Tensor-vs-standard had no top-1 mismatches, but reached
worst RMS `0.667784` and worst top20 abs `1.47467` at resumed frontier `1024`.
After switching coverage to candidate-vs-default-Tensor, the
`mpp-fast-continuation-chunks` candidate still fails: `tensor_vs_default_tensor`
worst RMS is `0.512339` at frontier `2048`, and worst top20 abs is `1.41916`
at frontier `1024`.

The local-run index now also picks up persistent chart-only runs from
`run_metal_tensor_bench.sh`, so the saved current-branch charts are visible
beside candidate gates, drift gates, comparator probes, and stage profiles.
For the latest chart run,
`20260515-052156-metal-tensor-bench`, Tensor prefill was `+15.1%..+31.4%`
versus standard Metal across the eight measured frontiers, while generation was
`-1.3%..-0.5%`.

## Experimental Routed-MoE Matmul Recheck

Rechecked the experimental routed-MoE matmul window on the current candidate
gate because the older notes had an under-verified start-layer 15 result. Both
runs used `--run-drift-gate --no-fail`, so drift would only run after the
speed screen passed.

Artifacts:

- `speed-bench/local-runs/20260515-080102-experimental-moe-matmul-start15-current/prefill-candidate-summary.md`
- `speed-bench/local-runs/20260515-080356-experimental-moe-matmul-start14-current/prefill-candidate-summary.md`
- `speed-bench/local-runs/20260515-080749-experimental-moe-matmul-gateup14-down12-current/prefill-candidate-summary.md`
- `speed-bench/local-runs/20260515-080658-local-run-index/local-run-index.md`
- `speed-bench/local-runs/20260515-081042-local-run-index/local-run-index.md`

Two-repeat median speed versus current Tensor default:

| Candidate | 512 | 1024 | 2048 | 4096 | 8192 | Min repeat prefill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1`, start layer `15` | -0.6% | -0.0% | +0.2% | +2.5% | +3.0% | -3.2% |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1`, start layer `14` | -0.6% | -0.5% | -0.7% | -0.8% | -0.2% | -2.1% |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1`, gate/up start layer `14`, down start layer `12` | -1.1% | -1.9% | -2.2% | -3.3% | -0.1% | -3.9% |

Conclusion: reject both before the five-fixture drift gate. Start layer 15 is
only useful at larger contexts and is not repeat-stable; start layer 14 is
slower at every compact prefill point; preserving the current down-from-12
window while moving gate/up to 14 is slower still. The current conservative
routed-MoE default remains the baseline.

## Current Prefill Frontier Audit

Regenerated the persistent current-branch standard/quality/Tensor chart with
`speed-bench/run_metal_tensor_bench.sh` after moving chart artifacts out of
`/tmp` and into ignored local storage.

Artifacts:

- `speed-bench/local-runs/20260515-083543-metal-tensor-bench/20260515-083543_gen128_ds4_bench_standard_quality_tensor.png`
- `speed-bench/local-runs/20260515-083543-metal-tensor-bench/20260515-083543_gen128_ds4_bench_standard_metal.csv`
- `speed-bench/local-runs/20260515-083543-metal-tensor-bench/20260515-083543_gen128_ds4_bench_quality.csv`
- `speed-bench/local-runs/20260515-083543-metal-tensor-bench/20260515-083543_gen128_ds4_bench_tensor_metal.csv`
- `speed-bench/local-runs/20260515-084949-local-run-index/local-run-index.md`

Latest chart result versus standard Metal:

| Context | Tensor prefill gain | Tensor generation gain |
| ---: | ---: | ---: |
| 512 | +35.6% | +0.1% |
| 1024 | +42.4% | +0.6% |
| 2048 | +34.6% | +0.4% |
| 4096 | +30.0% | +0.2% |
| 8192 | +23.5% | -0.3% |
| 16384 | +18.9% | -0.1% |
| 32768 | +18.8% | -0.3% |
| 65536 | +15.7% | -0.3% |

The local-run index now sees four persistent Metal Tensor chart runs and keeps
them beside candidate gates, drift gates, comparator probes, and stage
profiles.

Re-audited the current MoE dispatch path before starting another kernel probe:

- `ds4_gpu_routed_moe_batch_tensor()` already builds one expert-major route map
  and reuses it for gate, up, and down;
- the map stage is not the measured bottleneck in the routed-MoE stage
  profiles;
- the final `kernel_mul_mm_id` writeback is a real scatter through `hids`, not
  a dense store that can be replaced safely with a one-line `simdgroup_store`;
- already-rejected probes cover paired gate/up, `tiidx` writeback, direct
  down-sum, N64/tok64/row-pair dense Q8, F16 RHS, FlashAttention setup knobs,
  and route-window expansion.

Conclusion: the current default remains the production baseline because it has
the best confirmed low-drift envelope from the five-fixture gate. The next
prefill optimization should not be another env-only screen. It should be a
default-off kernel-family prototype, with routed MoE as the highest-value target
and dense Q8 as the secondary target:

1. Preserve the legacy simdgroup-MMA arithmetic/writeback order first.
2. Reduce real staging/writeback cost instead of just widening the existing
   cooperative-Tensor window.
3. Prove local comparator tightness on the touched route before speed gating.
4. Run `run_prefill_candidate_gate.py` speed-only first, then the five-fixture
   drift gate only after the speed floor passes.

## Rejected Routed-MoE Up-SwiGLU Fusion

Tried a bounded default-off routed-MoE prefill prototype that fused the legacy
`moe_up` grouped matmul with the SwiGLU/route-weight write into the `mid`
buffer. The idea was to keep the legacy simdgroup-MMA arithmetic for the up
projection while avoiding the up scratch write/read and separate activation
dispatch.

Initial speed artifact:

- `speed-bench/local-runs/20260515-085820-moe-prefill-up-swiglu/prefill-candidate-summary.md`

The speed-only part was promising versus the then-current Tensor baseline:

| Context | Candidate prefill vs Tensor | Candidate generation vs Tensor |
| ---: | ---: | ---: |
| 512 | +6.7% | -0.1% |
| 1024 | +37.7% | +0.5% |
| 2048 | +23.7% | +0.4% |
| 4096 | +14.3% | +0.0% |
| 8192 | +12.6% | +0.1% |

The first drift scorecard for that artifact was invalid because the helper had
rebuilt `ds4-bench` for the speed path but the drift gate used a stale `ds4`
binary. After rebuilding `ds4`/`ds4_test`, `./ds4_test --metal-mpp-equivalence`
with `DS4_METAL_MOE_PREFILL_UP_SWIGLU=1` failed hard on the long fixtures:

| Fixture | Same top1 | Top20 | RMS | Top20 abs | Greedy |
| --- | --- | ---: | ---: | ---: | --- |
| `long_memory_archive` | no | 12/20 | 1.80489 | 6.19391 | diff@0 |
| `long_code_audit` | no | 11/20 | 1.95671 | 4.80762 | diff@0 |

Setting `DS4_METAL_MOE_MID_F32=1` did not change the failure shape, so this is
not just the F16 mid storage path. The fused kernel/prototype was removed rather
than kept as another broken env mode.

Tooling fix from this miss:

- `run_quality_drift_gate.py` now refuses to run against a stale `ds4` binary
  when core sources or `metal/*.metal` are newer than the binary.
- `run_prefill_candidate_gate.py` now does the same for `ds4-bench` and passes
  the guard through to nested quality drift gates.
- `run_chunked_prefill_drift_gate.py` now applies the same stale-`ds4-bench`
  guard for standalone resumed-frontier coverage runs.
- `run_metal_tensor_bench.sh` now applies the same stale-`ds4-bench` guard for
  persistent standard/quality/Tensor chart regeneration.
- `run_mpp_compare_probe.py` now applies the same stale-`ds4` guard for local
  comparator probes.
- `--allow-stale-binary` exists only for intentional old-artifact summaries.

Fresh restored-baseline artifacts:

- `speed-bench/local-runs/20260515-091751-current-default-quality-drift-gate/summary.md`

The fresh no-env five-fixture gate is back to the known-good default envelope:
Tensor-vs-standard has top1 mismatches `0`, greedy mismatches `0`, min top20
`19/20`, worst RMS `0.239946`, and worst top20 abs `0.55422`.

## Rejected Narrow Gate/Up Route Windows

Screened the narrower routed-MoE gate/up Tensor window that was still adjacent
to the rejected `0-3` and `0-5` sweeps:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --candidate-label mpp-gateup0-1-down12 \
  --set-env DS4_METAL_MPP_MOE_GATE_START_LAYER=0 \
  --set-env DS4_METAL_MPP_MOE_GATE_FILTER=layer=0-1,layer=15-42 \
  --set-env DS4_METAL_MPP_MOE_UP_START_LAYER=0 \
  --set-env DS4_METAL_MPP_MOE_UP_FILTER=layer=0-1,layer=15-42 \
  --no-fail
```

Artifact:

- `speed-bench/local-runs/20260515-093425-mpp-gateup0-1-down12/prefill-candidate-summary.md`

Two-repeat median speed versus the current Tensor default:

| Context | Candidate prefill vs Tensor | Candidate generation vs Tensor |
| ---: | ---: | ---: |
| 512 | -0.4% | -0.6% |
| 1024 | -0.2% | -0.4% |
| 2048 | -0.7% | -0.2% |
| 4096 | +0.6% | -0.3% |
| 8192 | +2.2% | -0.1% |

The repeat-level floor also failed with min repeat prefill `-3.6%`. Reject
before drift gate: a two-layer early gate/up expansion only helps larger compact
contexts and still regresses the short/mid contexts.

Then screened the remaining `0-2` gap:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --candidate-label mpp-gateup0-2-down12 \
  --set-env DS4_METAL_MPP_MOE_GATE_START_LAYER=0 \
  --set-env DS4_METAL_MPP_MOE_GATE_FILTER=layer=0-2,layer=15-42 \
  --set-env DS4_METAL_MPP_MOE_UP_START_LAYER=0 \
  --set-env DS4_METAL_MPP_MOE_UP_FILTER=layer=0-2,layer=15-42 \
  --no-fail
```

Artifact:

- `speed-bench/local-runs/20260515-093802-mpp-gateup0-2-down12/prefill-candidate-summary.md`

Two-repeat median speed versus the current Tensor default:

| Context | Candidate prefill vs Tensor | Candidate generation vs Tensor |
| ---: | ---: | ---: |
| 512 | +2.2% | -0.0% |
| 1024 | +3.1% | +2.3% |
| 2048 | +2.0% | +0.4% |
| 4096 | +0.0% | -0.2% |
| 8192 | -0.7% | -0.1% |

The repeat-level floor failed with min repeat prefill `-2.0%`. Reject before
drift gate: it improves the short/mid contexts but gives back the 8192 point and
is not repeat-stable at 4096 or 8192. This closes the narrow route-window gap
between the failed `0-1`, repeat-unstable `0-3`, and slower `0-5` screens; route
window expansion remains exhausted.

## Rejected Routed-MoE X-F16 Prepack Probe

Tried a local default-off prototype, `DS4_METAL_MOE_PREFILL_X_F16=1`, that
prepacked the routed-MoE input activation to half once per layer and fed the
existing F16-RHS routed matmul variants for gate/up. The goal was to avoid
restaging the same F32 input as half separately in both gate and up matmuls
without changing the default path.

Artifact:

- `speed-bench/local-runs/20260515-094520-moe-prefill-x-f16/prefill-candidate-summary.md`

Two-repeat median speed versus the current Tensor default:

| Context | Candidate prefill vs Tensor | Candidate generation vs Tensor |
| ---: | ---: | ---: |
| 512 | -2.9% | +0.1% |
| 1024 | +0.2% | -0.4% |
| 2048 | +0.2% | +0.1% |
| 4096 | +0.5% | -0.2% |
| 8192 | +2.5% | -0.9% |

The repeat-level floor failed with min repeat prefill `-8.0%`, so the
five-fixture drift gate was not run. The copy/prepack cost is too high at short
contexts and too noisy through the compact gate. The prototype code was removed
rather than kept as another non-promotable environment mode.

Fresh restored-baseline check after removing the prototype:

- `speed-bench/local-runs/20260515-095024-current-default-quality-drift-gate/summary.md`

The no-env five-fixture gate passed. Tensor-vs-standard had top1 mismatches
`0`, greedy mismatches `0`, min top20 `19/20`, worst RMS `0.239946`, and worst
top20 abs `0.55422`, matching the known current-default envelope.

## Current-Default Residual `moe_down` Comparator

Ran a current-default local comparator on the `long_memory_archive` fixture to
attribute the remaining conservative Tensor-vs-standard movement before trying
another kernel candidate:

```sh
python3 speed-bench/run_mpp_compare_probe.py \
  --route moe_gate,moe_up,moe_down \
  --case long_memory_archive \
  --compare-max 120 \
  --continue-after-breach \
  --verbose
```

Artifact:

- `speed-bench/local-runs/20260515-095750-manual-mpp-compare-probe/mpp-compare-summary.md`

The current default still has clean local `moe_gate` and `moe_up` comparisons
under the `max_abs <= 0.001` target. All target breaches came from `moe_down`,
mostly in late layers. The worst local delta was `layer=42` with max abs
`0.0166016` and RMS `8.91692e-06`; the other breaches were layers `26`, `29`,
`30`, `31`, `32`, `33`, `34`, `35`, `36`, `37`, `38`, `39`, and `40`.

Repeated the same current-default comparator on `long_code_audit`, the fixture
responsible for current-default worst Tensor-vs-standard RMS in the five-case
gate:

- `speed-bench/local-runs/20260515-100424-manual-mpp-compare-probe/mpp-compare-summary.md`

The result matched `long_memory_archive`: 87 comparisons, the same 14 local
`moe_down` breaches, no `moe_gate`/`moe_up` target breach, and the same worst
layer-42 max abs `0.0166016` with RMS `8.37744e-06`.

Tried a local default-off implementation probe,
`DS4_METAL_MPP_MOE_DOWN_FAST_LAYOUT=0`, that disabled the first-PR fast MPP
layout only for `moe_down` while leaving gate/up on the current fast layout.
This was meant to test whether the late `moe_down` residual drift came from the
fast-layout staging/writeback instead of the cooperative Tensor matmul itself.

Artifact:

- `speed-bench/local-runs/20260515-100727-manual-mpp-compare-probe/mpp-compare-summary.md`

The comparator result was unchanged from the current default on
`long_code_audit`: 31 `moe_down` comparisons, the same 14 target breaches, and
the same worst layer-42 max abs `0.0166016` with RMS `8.37744e-06`. Reject and
remove the hook before speed/drift gates. The remaining `moe_down` movement is
not fixed by swapping the MPP fast layout for the generic MPP layout; it needs a
new arithmetic path, not a layout selector.

That suggested the only simple drift mitigation left for the promoted default
would be narrowing `moe_down` to the locally clean early range. Screened that
candidate without the drift gate:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --out-dir speed-bench/local-runs/20260515-095930-current-down12-25 \
  --candidate-label current-down12-25 \
  --set-env DS4_METAL_MPP_MOE_DOWN_FILTER=layer=12-25 \
  --no-fail
```

Artifact:

- `speed-bench/local-runs/20260515-095930-current-down12-25/prefill-candidate-summary.md`

Two-repeat median speed versus the current Tensor default:

| Context | Candidate prefill vs Tensor | Candidate generation vs Tensor |
| ---: | ---: | ---: |
| 512 | -4.9% | -0.0% |
| 1024 | -3.8% | +0.4% |
| 2048 | -2.6% | +1.5% |
| 4096 | -1.5% | +0.8% |
| 8192 | -3.1% | -1.1% |

The repeat-level floor also failed with min repeat prefill `-6.5%`. Reject
before drift gate: the current conservative default's residual local
`moe_down` movement is real, but disabling the late down Tensor layers gives up
too much prefill throughput. Do not spend more route-filter time on cleaning
current-default `moe_down` drift unless a new down kernel preserves the speed of
the late Tensor route.

Refreshed local run index after these artifacts:

- `speed-bench/local-runs/20260515-100856-local-run-index/local-run-index.md`

## Rejected Strict `mpp-fast` Route Window Recheck

Reran the earlier `mpp-fast` gate/up/down route-window candidate against the
current branch after the later drift and cleanup work, using the strict
repeat-floor candidate gate:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --out-dir speed-bench/local-runs/20260515-101058-mpp-fast-gate0-up15-down12-current-strict \
  --candidate-label mpp-fast-gate0-up15-down12-current-strict \
  --set-env DS4_METAL_MPP_FAST=1 \
  --set-env DS4_METAL_MPP_MOE_UP_START_LAYER=15 \
  --set-env DS4_METAL_MPP_MOE_DOWN_START_LAYER=12 \
  --run-drift-gate \
  --no-fail
```

Artifact:

- `speed-bench/local-runs/20260515-101058-mpp-fast-gate0-up15-down12-current-strict/prefill-candidate-summary.md`

Two-repeat median speed versus the current Tensor default:

| Context | Candidate prefill vs Tensor | Candidate generation vs Tensor |
| ---: | ---: | ---: |
| 512 | +3.6% | -0.3% |
| 1024 | +1.8% | -0.2% |
| 2048 | +2.5% | -0.1% |
| 4096 | +3.7% | -0.4% |
| 8192 | +4.4% | +0.3% |

Reject before drift gate. The median profile is useful, but the repeat-level
prefill floor failed with min repeat `-0.1%` at 1024 tokens, so it is not
promotion-stable under the strict gate. This keeps the current conservative
default as the baseline and leaves future work focused on a new routed-MoE
arithmetic path rather than more environment-only route-window tuning.

Refreshed local run index after this artifact:

- `speed-bench/local-runs/20260515-101358-local-run-index/local-run-index.md`

## Rejected Current-Default Gate/Up Layer-16 Contraction

Closed the one remaining small route-window gap around the current conservative
default by moving only gate/up from layer 15 to layer 16 while leaving down at
layer 12:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --out-dir speed-bench/local-runs/20260515-101837-mpp-gateup16-down12-current-strict \
  --candidate-label mpp-gateup16-down12-current-strict \
  --set-env DS4_METAL_MPP_MOE_GATE_START_LAYER=16 \
  --set-env DS4_METAL_MPP_MOE_UP_START_LAYER=16 \
  --set-env DS4_METAL_MPP_MOE_DOWN_START_LAYER=12 \
  --run-drift-gate \
  --no-fail
```

Artifact:

- `speed-bench/local-runs/20260515-101837-mpp-gateup16-down12-current-strict/prefill-candidate-summary.md`

Two-repeat median speed versus the current Tensor default:

| Context | Candidate prefill vs Tensor | Candidate generation vs Tensor |
| ---: | ---: | ---: |
| 512 | -2.6% | -0.2% |
| 1024 | -1.9% | -0.8% |
| 2048 | -1.7% | +0.1% |
| 4096 | -0.5% | -0.5% |
| 8192 | +1.0% | -0.4% |

Reject before drift gate. The contraction fails both the median prefill floor
and repeat-level floor, with min median prefill `-2.6%` and min repeat prefill
`-4.7%`. This confirms the current layer-15 gate/up window is still the better
production baseline; the next useful improvement remains a new default-off
routed-MoE arithmetic path rather than shifting the conservative route window.

Refreshed local run index after this artifact:

- `speed-bench/local-runs/20260515-102142-local-run-index/local-run-index.md`

## Rejected MoE `sum6` Vec4 Probe

Tried a local default-off probe, `DS4_METAL_MOE_SUM6_VEC4=1`, that replaced the
six-expert post-down summation kernel with a `float4` vectorized load/add/store
variant when `out_dim`, offsets, and strides were 16-byte aligned. This kept the
same expert summation order and did not change the grouped down matmul.

Artifact:

- `speed-bench/local-runs/20260515-102448-moe-sum6-vec4/prefill-candidate-summary.md`

Two-repeat median speed versus the current Tensor default:

| Context | Candidate prefill vs Tensor | Candidate generation vs Tensor |
| ---: | ---: | ---: |
| 512 | -2.2% | +0.1% |
| 1024 | -1.5% | -0.1% |
| 2048 | -2.0% | -0.2% |
| 4096 | -1.1% | -0.0% |
| 8192 | +1.6% | +0.1% |

Reject before drift gate. The median prefill floor failed with min `-2.2%`,
and the repeat-level floor failed with min repeat `-5.3%`. The temporary
kernel and environment hook were removed after the screen. The existing scalar
`sum6` kernel remains the baseline; optimizing the sum stage alone is not a
useful compact prefill path unless a future design also changes the down/sum
dataflow without losing expert-major matmul throughput.

Refreshed local run index after this artifact:

- `speed-bench/local-runs/20260515-102819-local-run-index/local-run-index.md`

## Rejected Strict MoE `sum6` Disable Recheck

Reran the older `DS4_METAL_MOE_SUM6_DISABLE=1` control through the current
strict two-repeat candidate gate. The earlier one-off control had shown a
small-context median gain, so this recheck tests whether that survives the
repeat-floor rule used for promotion.

Artifact:

- `speed-bench/local-runs/20260515-103032-disable-moe-sum6-current-strict/prefill-candidate-summary.md`

Two-repeat median speed versus the current Tensor default:

| Context | Candidate prefill vs Tensor | Candidate generation vs Tensor |
| ---: | ---: | ---: |
| 512 | -1.6% | +0.2% |
| 1024 | -2.0% | -0.3% |
| 2048 | -1.8% | -0.1% |
| 4096 | -2.0% | -1.0% |
| 8192 | +0.3% | +0.1% |

Reject before drift gate. The median prefill floor failed with min `-2.0%`,
and the repeat-level floor failed with min repeat `-5.3%`. Together with the
rejected vec4 probe, this closes the current `sum6` stage as a standalone
prefill optimization target. A future down/sum direction needs a different
dataflow, not another replacement for the final summation kernel.

Refreshed local run index after this artifact:

- `speed-bench/local-runs/20260515-103339-local-run-index/local-run-index.md`

## Current FlashAttention Stage Profile Refresh

Reran the isolated static-mixed FlashAttention stage profiler on the current
branch after the routed-MoE and `sum6` cleanup work. This was a profile-only
baseline, not a production candidate.

Command:

```sh
env DS4_METAL_FLASH_ATTN_STAGE_PROFILE=1 \
    DS4_METAL_FLASH_ATTN_STAGE_PROFILE_FILTER=static_mixed \
    ./ds4-bench -mt auto \
      --prompt-file speed-bench/promessi_sposi.txt \
      --ctx-start 2048 --ctx-max 2048 --gen-tokens 1 \
      --csv speed-bench/local-runs/20260515-103653-current-flash-attn-stage-profile-2048/bench.csv
```

Artifacts:

- `speed-bench/local-runs/20260515-103653-current-flash-attn-stage-profile-2048/bench.csv`
- `speed-bench/local-runs/20260515-103653-current-flash-attn-stage-profile-2048/stage-profile-summary.md`
- `speed-bench/local-runs/20260515-103653-current-flash-attn-stage-profile-2048/stage-profile-summary.json`

The measured 2048-token throughput was `471.50` prefill t/s and `35.92`
generation t/s. Parsed FlashAttention profile time was `506.613 ms` across
`225` events:

| Stage | total ms | events | share |
| --- | ---: | ---: | ---: |
| `flash_attn.static_mixed_nonvec.attention` | 425.729 | 41 | 84.0% |
| `flash_attn.static_mixed_nonvec.mask_fill` | 46.790 | 41 | 9.2% |
| `flash_attn.static_mixed_nonvec.block_map` | 10.250 | 41 | 2.0% |
| `flash_attn.static_mixed_nonvec.copy_raw` | 9.164 | 41 | 1.8% |
| `flash_attn.static_mixed_nonvec.copy_comp` | 8.179 | 41 | 1.6% |
| `flash_attn.static_mixed_nonvec.pad` | 6.501 | 20 | 1.3% |

Shape split:

| Shape | total ms | events |
| --- | ---: | ---: |
| `tokens=2048 comp=512 keys=2560 ratio=4` | 316.188 | 105 |
| `tokens=2048 comp=16 keys=2064 ratio=128` | 190.425 | 120 |

Conclusion: the current branch still matches the earlier FlashAttention triage.
The isolated attention kernel body dominates the FlashAttention slice, while
the full current `promessi_sposi` stage profile shows that slice is only a
secondary whole-model prefill target (`0.7%` parsed stage share for
`flash_attn.static_mixed_nonvec.attention`). Keep FlashAttention deprioritized
unless the next pass is a true static-mixed-specific kernel family with local
head-output comparison; do not repeat the already rejected setup/mask/tile
knobs.

Refreshed local run index after this artifact:

- `speed-bench/local-runs/20260515-103729-local-run-index/local-run-index.md`

## Rejected Current-Default F32-Mid `moe_down` Comparator Check

Ran a current-default `moe_down` local comparator with
`DS4_METAL_MOE_MID_F32=1` on `long_code_audit` to check whether the residual
late-layer `moe_down` movement came from the F16 routed-MoE intermediate rather
than the Tensor matmul route.

Command:

```sh
python3 speed-bench/run_mpp_compare_probe.py \
  --out-dir speed-bench/local-runs/20260515-103935-current-mid-f32-moe-down-compare \
  --route moe_down \
  --case long_code_audit \
  --compare-max 120 \
  --continue-after-breach \
  --verbose \
  --set-env DS4_METAL_MOE_MID_F32=1
```

Artifact:

- `speed-bench/local-runs/20260515-103935-current-mid-f32-moe-down-compare/mpp-compare-summary.md`

Result: unchanged from the no-env current-default comparator. The probe parsed
`31` `moe_down` comparisons and found the same `14` target breaches. Worst
delta remained layer 42 with max abs `0.0166016` and RMS `8.37744e-06`.

Conclusion: reject before speed or five-fixture drift gates. Keeping the MoE
intermediate in F32 does not clean up the current default's local `moe_down`
movement, so the remaining residual is still in the routed Tensor matmul
arithmetic path rather than the F16 mid buffer.

## Attention-Output Stage Profiler Boundary Fix

Tried a focused attention-output stage profile to split the promoted
attention-output route into its low projection and final Q8 output projection:

- initial artifact:
  `speed-bench/local-runs/20260515-104057-current-attn-out-stage-profile-2048/stage-profile-summary.md`

The first run exposed a profiler issue rather than a kernel result:
`attn_output.low_proj` reported `3778.693 ms` total (`87.877 ms` per layer),
which was inconsistent with the full-model profile. The attention-output
profiler did not flush the pending command buffer at function entry, so the
first `low_proj` timing in each layer included upstream queued work.

Patch: make `DS4_METAL_ATTN_OUT_STAGE_PROFILE=1` follow the MoE and
FlashAttention profiler pattern by ending the current batch and starting a new
command buffer before starting the first attention-output stage timer. This is
profiling-only code; normal inference is unchanged unless the profile env is
set.

Validation:

```sh
make ds4-bench ds4_test ds4
```

Fixed-profile artifact:

- `speed-bench/local-runs/20260515-104146-current-attn-out-stage-profile-2048/stage-profile-summary.md`

Fixed 2048-token profile:

| Stage | total ms | events | avg ms | share |
| --- | ---: | ---: | ---: | ---: |
| `attn_output.out_proj` | 441.999 | 43 | 10.279 | 41.2% |
| `q8.attn_out` | 436.981 | 43 | 10.162 | 40.7% |
| `attn_output.low_proj` | 195.033 | 43 | 4.536 | 18.2% |

Conclusion: the promoted attention-output low projection is no longer the
dominant target in this route. The remaining secondary hotspot is the final
generic Q8 `attn_out` output projection. That keeps dense Q8 as the secondary
kernel-family target, but the already rejected Q8 tile/direct-RHS/row-pair
probes still apply; a future attempt needs a genuinely new out-projection Q8
kernel design, not another host-side profiler or tile switch.

Refreshed local run index after these artifacts:

- `speed-bench/local-runs/20260515-104232-local-run-index/local-run-index.md`

## Current Default Drift Gate After Profiler Fix

Reran the no-env five-fixture quality drift gate after the
attention-output profiler boundary fix and rebuild. The profiler fix is gated
behind `DS4_METAL_ATTN_OUT_STAGE_PROFILE`, but this refresh keeps the branch
evidence current after touching `ds4_metal.m`.

Command:

```sh
python3 speed-bench/run_quality_drift_gate.py \
  --no-fail \
  --out-dir speed-bench/local-runs/20260515-104329-current-default-quality-drift-gate
```

Artifacts:

- `speed-bench/local-runs/20260515-104329-current-default-quality-drift-gate/summary.md`
- `speed-bench/local-runs/20260515-104329-current-default-quality-drift-gate/summary.json`

Gate result: `OK`.

| Pair | Top1 mismatches | Greedy mismatches | Min top20 | Worst RMS | Worst top20 abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `standard_vs_quality` | 0 | 1 | 18/20 | 0.618172 | 2.24006 |
| `tensor_vs_quality` | 0 | 1 | 18/20 | 0.618172 | 2.24006 |
| `tensor_vs_standard` | 0 | 0 | 19/20 | 0.239946 | 0.55422 |

Conclusion: current default Tensor remains in the established low-drift
envelope after the profiler-only code change.

Refreshed local run index after this artifact:

- `speed-bench/local-runs/20260515-104628-local-run-index/local-run-index.md`

## Routed-MoE Down/Sum Follow-Up Boundary

Follow-up code inspection after the current-default `moe_down` comparator
checks and the attention-output profiler fix. This does not reopen the older
rejected `DS4_METAL_MOE_PREFILL_DIRECT_DOWN_SUM=1` prototype; that artifact
was already strongly negative:

- `speed-bench/local-runs/20260515-044921-moe-prefill-direct-down-sum/prefill-candidate-summary.md`
  (`-19.7%`, `-20.1%`, `-29.6%` prefill at 512/1024/2048 vs Tensor).

Relevant current path shape:

- `kernel_mul_mm_id_map0` builds an expert-major token map (`htpe`/`hids`) so
  each routed matmul tile reuses one expert's weight rows across the tokens
  routed to that expert.
- `kernel_mul_mm_id` then writes each selected expert result into the
  token-major expert slot layout, and `kernel_dsv4_moe_sum6_f32` performs the
  final six-expert reduction.
- The measured `sum` stage is small compared with the matmuls
  (`~0.5-1.1 ms/layer` in the 2048/3844-token profiles), while `moe_down`
  itself is still one of the dominant stages.

Conclusion: a naive direct token-major down/sum kernel is closed. It loops over
six experts inside each output tile, removes useful expert-parallel work, and
attacks a small standalone sum cost while losing the grouped prefill matmul.
The next routed-MoE candidate should instead keep the expert-major map and
either:

1. introduce a reference-compatible early-window matmul variant that reduces
   staging/pointer overhead while preserving the legacy simdgroup-MMA arithmetic
   order, or
2. design a down/sum fused kernel that still dispatches expert-major work and
   only changes the final accumulation dataflow after a local `moe_down`
   comparator proves it is tight.

Acceptance remains unchanged: default-off env hook, local route comparator,
speed-only compact gate, then the five-fixture drift gate.

## Rejected Routed-MoE `ne20=6` Legacy Specialization

Tried a local default-off prototype, `DS4_METAL_MOE_NE20_6=1`, that
compile-time-specialized the legacy routed-MoE `kernel_mul_mm_id` path for the
DS4 fixed six selected experts. The prototype preserved the existing legacy
simdgroup-MMA arithmetic path and only replaced runtime `args.ne20` division and
modulo with a template constant for the early non-MPP routed-MoE matmuls.

Local comparator smoke:

- `speed-bench/local-runs/20260515-151302-moe-ne20-6-compare-long-code/mpp-compare-summary.md`

The comparator parsed `129` route comparisons on `long_code_audit`. `moe_gate`
and `moe_up` stayed under target. The only breaches were the already-known late
`moe_down` Tensor residuals, with the same worst layer-42 max abs `0.0166016`
and RMS `8.37744e-06`.

Speed artifact:

- `speed-bench/local-runs/20260515-151422-moe-ne20-6/prefill-candidate-summary.md`

Two-repeat median speed versus the current Tensor default:

| Context | Candidate prefill vs Tensor | Candidate generation vs Tensor |
| ---: | ---: | ---: |
| 512 | +1.1% | +0.1% |
| 1024 | +2.2% | -0.1% |
| 2048 | +1.7% | -1.4% |
| 4096 | +0.0% | -1.0% |
| 8192 | +1.4% | -0.1% |

Reject before drift gate. The median line is mildly positive, but the strict
repeat floor failed with min repeat prefill `-4.0%` and min repeat generation
`-2.6%`. This is too small and noisy to keep as another default-off production
path. The prototype code was removed after the screen.

Refreshed local run index after this artifact:

- `speed-bench/local-runs/20260515-152039-local-run-index/local-run-index.md`

## Rejected Narrow Continuation-Chunk Early MoE Window

Screened a narrower version of the earlier continuation-chunk idea using the
existing `module@layer` filter syntax. This kept the current conservative
`pos=0` defaults, then added only routed-MoE layers `0..3` on resumed
frontiers `512`, `1024`, `2048`, and `4096`:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --out-dir speed-bench/local-runs/20260515-152507-mpp-cont-gud0-3 \
  --candidate-label mpp-cont-gud0-3 \
  --set-env DS4_METAL_MPP_FAST=1 \
  --set-env 'DS4_METAL_MPP_MOE_GATE_FILTER=layer=15-42,pos=512 routed_moe@layer=0-3,pos=1024 routed_moe@layer=0-3,pos=2048 routed_moe@layer=0-3,pos=4096 routed_moe@layer=0-3' \
  --set-env 'DS4_METAL_MPP_MOE_UP_FILTER=layer=15-42,pos=512 routed_moe@layer=0-3,pos=1024 routed_moe@layer=0-3,pos=2048 routed_moe@layer=0-3,pos=4096 routed_moe@layer=0-3' \
  --set-env 'DS4_METAL_MPP_MOE_DOWN_FILTER=layer=12-42,pos=512 routed_moe@layer=0-3,pos=1024 routed_moe@layer=0-3,pos=2048 routed_moe@layer=0-3,pos=4096 routed_moe@layer=0-3' \
  --run-drift-gate \
  --no-fail
```

Artifact:

- `speed-bench/local-runs/20260515-152507-mpp-cont-gud0-3/prefill-candidate-summary.md`

Two-repeat median speed versus the current Tensor default:

| Context | Candidate prefill vs Tensor | Candidate generation vs Tensor |
| ---: | ---: | ---: |
| 512 | -1.7% | +0.3% |
| 1024 | +2.4% | -0.3% |
| 2048 | +0.4% | -0.4% |
| 4096 | +1.5% | -0.3% |
| 8192 | +1.9% | -0.6% |

Reject before drift gate. The median line was weakly positive after the first
frontier, but the strict speed screen failed with min median prefill `-1.7%`
and min repeat prefill `-5.8%`. This makes the narrow continuation route too
noisy to pursue into chunked drift coverage.

Refreshed local run index after this artifact:

- `speed-bench/local-runs/20260515-152840-local-run-index/local-run-index.md`

## Rejected Dense Q8 Half-Dequant Probe

Tried a local default-off prototype, `DS4_METAL_Q8_HALF_DEQUANT=1`, that kept
the existing dense Q8 prefill tile shape but dequantized the packed Q8 blocks
through `half` values instead of the existing float temporary path.

Local comparator smokes:

- `speed-bench/local-runs/20260515-153048-q8-half-dequant-compare/mpp-compare-summary.md`
- `speed-bench/local-runs/20260515-153048-q8-half-dequant-compare-attn-out/mpp-compare-summary.md`

Both comparator smokes parsed `3` Q8 comparisons and found exact zero deltas
for their filtered early-layer checks:

- `attn_q_b`: worst max abs `0`, RMS `0`
- `attn_out`: worst max abs `0`, RMS `0`

Speed artifact:

- `speed-bench/local-runs/20260515-153122-q8-half-dequant/prefill-candidate-summary.md`

Two-repeat median speed versus the current Tensor default:

| Context | Candidate prefill vs Tensor | Candidate generation vs Tensor |
| ---: | ---: | ---: |
| 512 | -5.6% | -2.1% |
| 1024 | -9.0% | -4.2% |
| 2048 | -6.8% | -2.3% |
| 4096 | -4.4% | +0.1% |
| 8192 | -0.2% | +0.1% |

Reject before drift gate. The local comparator was exact on the two smoke
routes, but the speed screen failed badly: min median prefill was `-9.0%` and
min repeat prefill was `-13.5%`. The prototype code was removed after the
screen.

## Refreshed Persistent Metal Tensor Bench Chart

Regenerated the current branch Standard Metal / Quality Metal / Tensor Metal
chart using:

```sh
OPEN_CHART=0 speed-bench/run_metal_tensor_bench.sh
```

Artifacts:

- `speed-bench/local-runs/20260515-153948-metal-tensor-bench/20260515-153948_gen128_ds4_bench_quality.csv`
- `speed-bench/local-runs/20260515-153948-metal-tensor-bench/20260515-153948_gen128_ds4_bench_standard_metal.csv`
- `speed-bench/local-runs/20260515-153948-metal-tensor-bench/20260515-153948_gen128_ds4_bench_tensor_metal.csv`
- `speed-bench/local-runs/20260515-153948-metal-tensor-bench/20260515-153948_gen128_ds4_bench_standard_quality_tensor.png`

The artifacts live under `speed-bench/local-runs/`, which is ignored by
`speed-bench/.gitignore`, so repeated timestamped charts stay local.

| Context | Tensor prefill vs Standard | Tensor generation vs Standard | Quality prefill vs Standard |
| ---: | ---: | ---: | ---: |
| 512 | +34.6% | +1.5% | +3.9% |
| 1024 | +36.3% | +1.9% | +17.8% |
| 2048 | +31.0% | +2.4% | +12.1% |
| 4096 | +26.7% | +2.2% | +10.8% |
| 8192 | +25.0% | +1.9% | +5.7% |
| 16384 | +22.8% | +0.3% | -9.4% |
| 32768 | +19.3% | -0.0% | -3.7% |
| 65536 | +14.9% | -1.4% | -6.3% |

Current persistent chart summary: Tensor prefill remains ahead of Standard by
`+14.9%..+36.3%`; Tensor generation is roughly flat at `-1.4%..+2.4%`.

Refreshed local run index after these artifacts:

- `speed-bench/local-runs/20260515-155451-local-run-index/local-run-index.md`

## Current Default Drift Refresh After Chart Persistence

Reran the no-env five-fixture quality drift gate after the benchmark chart
script started writing timestamped artifacts under ignored `speed-bench/local-runs/`.
The first sandboxed attempt could not access the Metal device; the same command
was rerun with local Metal access:

```sh
python3 speed-bench/run_quality_drift_gate.py \
  --no-fail \
  --out-dir speed-bench/local-runs/20260515-171007-current-default-quality-drift-refresh
```

Artifacts:

- `speed-bench/local-runs/20260515-171007-current-default-quality-drift-refresh/summary.md`
- `speed-bench/local-runs/20260515-171007-current-default-quality-drift-refresh/summary.json`

Gate result: `OK`.

| Pair | Top1 mismatches | Greedy mismatches | Min top20 | Worst RMS | Worst top20 abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `standard_vs_quality` | 0 | 1 | 18/20 | 0.618172 | 2.24006 |
| `tensor_vs_quality` | 0 | 1 | 18/20 | 0.618172 | 2.24006 |
| `tensor_vs_standard` | 0 | 0 | 19/20 | 0.239946 | 0.55422 |

Conclusion: the current default Tensor route still matches the established
low-drift envelope while keeping the persistent benchmark artifacts local.

Refreshed local run index after this artifact:

- `speed-bench/local-runs/20260515-171500-local-run-index/local-run-index.md`

## AIME25 Eval Check

User-reported AIME25 eval result on the current baseline using the
`q2-imatrix` model:

| Mode | AIME25 score |
| --- | ---: |
| Standard Metal (`q2-imatrix`) | 86.7% |
| Tensor Metal (`q2-imatrix`) | 86.7% |

Conclusion: the current Tensor Metal baseline is quality-neutral on this eval
relative to Standard Metal, while retaining the measured prefill speed gain and
the clean five-fixture drift gate above.

## Current 8192-Context Stage Profile Refresh

Reran a focused current-default profile on the bench prompt at the 8192 context
row with layer, routed-MoE, Q8, FlashAttention, and attention-output stage
profiling enabled:

```sh
env DS4_METAL_LAYER_STAGE_PROFILE=1 \
    DS4_METAL_MOE_STAGE_PROFILE=1 \
    DS4_METAL_Q8_PREFILL_PROFILE=1 \
    DS4_METAL_FLASH_ATTN_STAGE_PROFILE=1 \
    DS4_METAL_ATTN_OUT_STAGE_PROFILE=1 \
    ./ds4-bench \
      --prompt-file speed-bench/promessi_sposi.txt \
      --ctx-start 8192 \
      --ctx-max 8192 \
      --gen-tokens 16 \
      --csv speed-bench/local-runs/20260515-155652-current-ctx8192-stage-profile/bench.csv
```

Artifacts:

- `speed-bench/local-runs/20260515-155652-current-ctx8192-stage-profile/bench.csv`
- `speed-bench/local-runs/20260515-155652-current-ctx8192-stage-profile/profile.stderr`
- `speed-bench/local-runs/20260515-155652-current-ctx8192-stage-profile/stage-profile-summary.md`
- `speed-bench/local-runs/20260515-155652-current-ctx8192-stage-profile/stage-profile-summary.json`

The profiled row measured `428.85` prefill tokens/s and `32.69` generation
tokens/s for the single 8192-context run. Parsed profile highlights:

| Stage | total ms | share |
| --- | ---: | ---: |
| `ffn.routed_moe` | 5802.228 | 17.7% |
| `attn.attention` | 4358.051 | 13.3% |
| `attn.output_proj` | 2468.958 | 7.5% |
| `attn.q_path` | 2439.041 | 7.4% |
| `moe_stage.up` | 1906.220 | 5.8% |
| `moe_stage.gate` | 1905.542 | 5.8% |
| `moe_stage.down` | 1735.243 | 5.3% |
| `q8.attn_out` | 1699.754 | 5.2% |
| `q8.attn_q_b` | 1682.686 | 5.1% |

MoE mask split:

| MoE mask | top stages | total ms |
| --- | --- | ---: |
| `0/0/0` | `gate`=859.1, `up`=855.5, `down`=852.5 | 2639.113 |
| `1/1/1` | `up`=837.2, `gate`=834.0, `down`=798.2 | 2626.682 |
| `0/0/1` | `up`=213.6, `gate`=212.5, `down`=84.6 | 527.369 |

Conclusion: dense Q8 `attn_q_b`/`attn_out` remain the largest non-MoE matmuls,
but the corrected generic Q8 MPP route and later Q8 probes are already closed
as slower. The bigger actionable bucket is still early routed-MoE work: the
legacy `0/0/0` layers cost about the same total time as the larger fully-Tensor
`1/1/1` window despite covering fewer events. Any new env screen should target
that early MoE region and must pass the five-fixture drift gate.

## Rejected Sparse Early Gate/Up Tensor Window

Screened a sparse early routed-MoE Tensor window based on the 8192-context
profile. The candidate left the current conservative `down` route unchanged
and added Tensor `gate`/`up` on early even layers `0,2,4,6,8,10` plus the
current default `15..42` range:

```sh
python3 speed-bench/run_prefill_candidate_gate.py \
  --out-dir speed-bench/local-runs/20260515-161513-mpp-gateup-even0-10-down12 \
  --candidate-label mpp-gateup-even0-10-down12 \
  --set-env DS4_METAL_MPP_MOE_GATE_START_LAYER=0 \
  --set-env DS4_METAL_MPP_MOE_GATE_FILTER=layer=0,layer=2,layer=4,layer=6,layer=8,layer=10,layer=15-42 \
  --set-env DS4_METAL_MPP_MOE_UP_START_LAYER=0 \
  --set-env DS4_METAL_MPP_MOE_UP_FILTER=layer=0,layer=2,layer=4,layer=6,layer=8,layer=10,layer=15-42 \
  --run-drift-gate \
  --no-fail
```

Artifact:

- `speed-bench/local-runs/20260515-161513-mpp-gateup-even0-10-down12/prefill-candidate-summary.md`

Two-repeat median speed versus the current Tensor default:

| Context | Candidate prefill vs Tensor | Candidate generation vs Tensor |
| ---: | ---: | ---: |
| 512 | +3.5% | +0.2% |
| 1024 | +4.1% | +0.0% |
| 2048 | +3.5% | -0.2% |
| 4096 | +4.2% | +0.2% |
| 8192 | +3.4% | -0.9% |

The speed signal was repeat-stable enough to run the five-fixture drift gate,
but the gate failed:

| Pair | Top1 mismatches | Greedy mismatches | Min top20 | Worst RMS | Worst top20 abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `standard_vs_quality` | 0 | 1 | 18/20 | 0.618172 | 2.24006 |
| `tensor_vs_quality` | 1 | 2 | 17/20 | 0.618172 | 2.45835 |
| `tensor_vs_standard` | 1 | 1 | 17/20 | 0.525365 | 2.47542 |

Reject. The prefill win is real, but the candidate introduces a top-1 mismatch
on `long_memory_archive`, a Tensor-vs-standard greedy mismatch, and a large
`long_code_audit` top20 drift. This is outside the branch's current low-drift
envelope.

Follow-up narrowed the sparse window to layers `4,6,8,10` only:

- `speed-bench/local-runs/20260515-162057-mpp-gateup-even4-10-down12/prefill-candidate-summary.md`

Two-repeat median speed versus the current Tensor default:

| Context | Candidate prefill vs Tensor | Candidate generation vs Tensor |
| ---: | ---: | ---: |
| 512 | +2.2% | -0.1% |
| 1024 | +3.1% | -0.7% |
| 2048 | +0.6% | -0.6% |
| 4096 | -0.6% | -0.8% |
| 8192 | +0.1% | +0.9% |

Reject before drift gate. Removing layers `0` and `2` avoids spending more
drift time, but it also loses the speed signal: min median prefill was `-0.6%`
and min repeat prefill was `-2.6%`. The sparse early-layer result therefore
does not expose a promotable speed/drift middle ground.

Refreshed local run index after these artifacts:

- `speed-bench/local-runs/20260515-162432-local-run-index/local-run-index.md`

## Rejected Early Gate/Up Parity Follow-Ups

Followed up the sparse even-layer result by splitting the early routed-MoE
gate/up additions into the `0,2` and odd-layer halves. Both candidates kept the
current conservative `down` route unchanged and only added Tensor `gate`/`up`
before the default `15..42` gate/up window.

### Layers `0,2`

Artifact:

- `speed-bench/local-runs/20260515-162536-mpp-gateup-even0-2-down12/prefill-candidate-summary.md`

Two-repeat median speed versus the current Tensor default:

| Context | Candidate prefill vs Tensor | Candidate generation vs Tensor |
| ---: | ---: | ---: |
| 512 | -2.0% | -0.7% |
| 1024 | -4.5% | -1.7% |
| 2048 | -2.3% | -1.0% |
| 4096 | +0.0% | -0.7% |
| 8192 | +2.6% | +0.7% |

Reject before drift gate. The isolated `0,2` window was slower through the
compact range, with min median prefill `-4.5%` and min repeat prefill `-6.8%`.

### Odd Layers `1,3,5,7,9,11`

Artifact:

- `speed-bench/local-runs/20260515-162841-mpp-gateup-odd1-11-down12/prefill-candidate-summary.md`

Two-repeat median speed versus the current Tensor default:

| Context | Candidate prefill vs Tensor | Candidate generation vs Tensor |
| ---: | ---: | ---: |
| 512 | +3.4% | -1.4% |
| 1024 | +2.2% | -0.8% |
| 2048 | +3.9% | -1.1% |
| 4096 | +1.6% | -0.3% |
| 8192 | +2.4% | -0.3% |

The speed screen passed, so the five-fixture drift gate ran:

| Pair | Top1 mismatches | Greedy mismatches | Min top20 | Worst RMS | Worst top20 abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `standard_vs_quality` | 0 | 1 | 18/20 | 0.618172 | 2.24006 |
| `tensor_vs_quality` | 0 | 1 | 17/20 | 0.618172 | 2.24006 |
| `tensor_vs_standard` | 0 | 0 | 17/20 | 0.54454 | 0.949314 |

Reject. The odd-layer sparse route is cleaner than the even `0,2,4,6,8,10`
screen because it introduces no top-1 or greedy mismatch, but the local
Tensor-vs-standard envelope is still too wide: RMS `0.54454` on
`long_memory_archive` and top20 abs `0.949314` on `long_code_audit`.

Conclusion for this direction: sparse early gate/up windows can buy another
`~2-4%` compact prefill, but the only speed-positive variants widen
Tensor-vs-standard drift well beyond the current branch envelope. This closes
the parity-shaped early-window idea unless a new arithmetic path reduces the
routed-MoE Tensor local movement.

Refreshed local run index after these artifacts:

- `speed-bench/local-runs/20260515-163440-local-run-index/local-run-index.md`

## Early Odd Gate/Up Drift Isolation

Followed the rejected `1,3,5,7,9,11` sparse gate/up candidate with a local
MoE comparator probe and two five-fixture drift splits. The goal was to check
whether the full-logit drift came from an obviously bad Tensor matmul site or
from cumulative early-layer movement.

Local comparator artifact:

- `speed-bench/local-runs/20260515-163903-manual-mpp-compare-probe/mpp-compare-summary.md`

The probe reused the rejected odd candidate filters and compared `moe_gate` and
`moe_up` separately on the two fixtures that drove the full-logit rejection:
`long_memory_archive` and `long_code_audit`.

| Metric | Value |
| --- | ---: |
| Parsed comparisons | 136 |
| Target breaches | 0 |
| Worst `moe_gate` max abs | 9.15527e-05 |
| Worst `moe_gate` RMS | 2.10598e-06 |
| Worst `moe_up` max abs | 9.91821e-05 |
| Worst `moe_up` RMS | 1.6725e-06 |

This clears the individual gate/up Tensor matmuls at the local comparator
threshold. The full-model drift is therefore not explained by a single bad
gate/up projection; it is more consistent with cumulative amplification from
moving early routed-MoE projections onto the Tensor path.

Then split the odd early window into `1,3,5` and `7,9,11`, keeping the current
default `down` route unchanged and retaining the default `15..42` gate/up
window.

### Layers `1,3,5`

Artifact:

- `speed-bench/local-runs/20260515-164155-drift-gate-gateup-odd1-5-down12/summary.md`

Tensor-vs-standard five-fixture result:

| Top1 mismatches | Greedy mismatches | Min top20 | Worst RMS | Worst top20 abs |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 19/20 | 0.569373 | 1.95196 |

Reject. This half keeps top-1 and greedy stable, but it fails the current
Tensor-vs-standard envelope on `long_memory_archive`: RMS `0.569373` and
top20 abs `1.95196`.

### Layers `7,9,11`

Artifact:

- `speed-bench/local-runs/20260515-164507-drift-gate-gateup-odd7-11-down12/summary.md`

Tensor-vs-standard five-fixture result:

| Top1 mismatches | Greedy mismatches | Min top20 | Worst RMS | Worst top20 abs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 16/20 | 0.518334 | 1.67467 |

Reject. This half is worse qualitatively: it introduces a top-1 and greedy
mismatch on `long_memory_archive`, and its worst RMS/top20 drift lands on
`long_code_audit`.

Conclusion: the speed-positive early odd gate/up window cannot be narrowed into
a safe half-window with the current Tensor arithmetic. Since both halves fail
the five-scenario drift gate, further speed benchmarking of these split windows
is not useful. Keep the promoted conservative route and do not add early
gate/up layers unless the underlying routed-MoE Tensor arithmetic changes.

Refreshed local run index after these artifacts:

- `speed-bench/local-runs/20260515-164718-local-run-index/local-run-index.md`

## Routed-MoE Kernel Variant Triage Refresh

Re-inspected the currently wired routed-MoE and attention-output Tensor
matmul variants after closing the sparse early-layer screens:

- `metal/moe.metal`: `kernel_mul_mm_id`, the generic MPP function-constant
  branch inside it, `kernel_mul_mm_id_mpp_fast_layout`,
  `kernel_mul_mm_id_pair_mpp`, and the attention-output low-Q8 MPP direct-RHS
  kernels.
- `ds4_metal.m`: `ds4_gpu_routed_mm_pipeline`,
  `ds4_gpu_routed_mm_f16_rhs_pipeline`, `ds4_gpu_encode_mul_mm_id_mapped_tile`,
  `ds4_gpu_encode_mul_mm_id_pair_mpp`, and the attention-output low-projection
  dispatch.

Status of the existing variants:

| Variant | Current status |
| --- | --- |
| Attention-output low-Q8 direct RHS | Promoted default; all-layer route passed the five-fixture gate and is part of the current baseline. |
| Attention-output staged RHS / tile-32 | Rejected as slower; keep direct RHS and tile-64 defaults. |
| Routed-MoE first-PR fast layout | Promoted only in the conservative layer window; wider early use is fast but widens Tensor-vs-standard drift. |
| Routed-MoE generic MPP function-constant path | Already screened via `DS4_METAL_MPP_MOE_FAST_LAYOUT=0`; it gives up speed without improving full-model drift. |
| Routed-MoE gate/up pair MPP | Rejected as consistently slower on both the old and current conservative windows. |
| Routed-MoE tile-64 | Rejected as slower. |

This leaves no untried source-level switch in the current routed-MoE Tensor
family that is likely to improve the prefill/drift tradeoff. The local
comparator shows individual early gate/up Tensor matmuls are clean at about
`1e-4` max abs, but five-fixture full-logit gates still fail when those early
layers are enabled. That points to cumulative arithmetic movement rather than
a single broken projection.

Next useful kernel work should be a new arithmetic-preserving routed-MoE
matmul path: keep the legacy simdgroup-MMA accumulation order as close as
possible, then optimize map/output overhead or memory layout around it. Another
`DS4_METAL_MPP_*` layer-window, tile-size, fast-layout, or pair-dispatch sweep
is unlikely to produce a promotable low-drift prefill win without changing the
underlying arithmetic.

## Rejected Routed-MoE Writeback Offset Simplification

Tried a local default-on source patch to simplify the final
`kernel_mul_mm_id` scatter address. The expert-major map stores each selected
output slot as `id = token * selected_experts + selected_slot`; in the current
host call shapes `args.ne1 == args.ne20`, so the writeback can algebraically
use `id * args.ne0` instead of recomputing `id % args.ne20` and
`id / args.ne20`.

This preserved the dequantization, simdgroup-MMA accumulation order, route
selection, and destination layout. It only changed the final destination pointer
calculation, with a fallback for the general `args.ne1 != args.ne20` case.

Artifacts:

- Baseline CSV:
  `speed-bench/local-runs/20260515-165545-pre-scatter-offset-baseline/tensor.csv`
- Patched CSV:
  `speed-bench/local-runs/20260515-165545-scatter-offset-patch/tensor.csv`

One compact `-mt auto` timing run versus the pre-patch source:

| Context | Prefill delta | Generation delta |
| ---: | ---: | ---: |
| 512 | -4.8% | +0.1% |
| 1024 | +0.3% | -0.2% |
| 2048 | +0.1% | -0.3% |
| 4096 | -0.4% | +0.5% |
| 8192 | -4.5% | +0.4% |

Reject before drift gate. The change is algebraically safe, but it did not
produce a speed signal and regressed the smallest and largest compact prefill
points in the smoke run. The patch was reverted and the binaries rebuilt from
the reverted source. Keep the existing writeback code unless a larger
source-level rewrite can remove more than this address arithmetic.

Refreshed local run index after this artifact:

- `speed-bench/local-runs/20260515-165926-local-run-index/local-run-index.md`
