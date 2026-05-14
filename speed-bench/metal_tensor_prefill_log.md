# Metal Tensor Prefill Optimization Log

Branch: `metal-tensor-prefill-quality-drift`

Date: 2026-05-14

This branch keeps the current low-drift Tensor default and uses the five-fixture
quality gate before promoting any prefill optimization.

## Drift Gate

Run:

```sh
python3 speed-bench/run_quality_drift_gate.py \
  --out-dir speed-bench/local-runs/20260514-1215-default-moe-19-19-19-quality-drift
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
| tensor vs standard | 0 | 0 | 0.136143 | 0.315292 |

Gate status: OK.

The direct equivalence test also passed:

```sh
./ds4_test --metal-mpp-equivalence
```

Result: `top1_mismatch=0`, `greedy_fail=0`, `worst_rms=0.136143`,
`worst_top20_max_abs=0.315292`.

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
  OUT_DIR=speed-bench/local-runs/20260514-1235-default-19-19-19-compact \
  OPEN_CHART=0 \
  speed-bench/run_metal_tensor_bench.sh
```

Current 19/19/19 Tensor default vs standard Metal:

| ctx | standard prefill t/s | tensor prefill t/s | tensor gain | standard gen t/s | tensor gen t/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 267.21 | 334.64 | 25.2% | 38.15 | 38.22 |
| 1024 | 272.68 | 337.80 | 23.9% | 37.94 | 37.05 |
| 2048 | 330.41 | 393.48 | 19.1% | 37.40 | 36.94 |
| 4096 | 341.26 | 386.55 | 13.3% | 34.31 | 34.11 |
| 8192 | 356.22 | 397.82 | 11.7% | 33.56 | 32.95 |

This keeps the plan focused on prefill. Generation is essentially unchanged.

## Rejected Knobs

These were evaluated as env-only candidates and not promoted.

| Candidate | Speed result | Drift result | Decision |
| --- | --- | --- | --- |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=18` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=18` | One run showed +2.2% to +5.7% over Tensor auto, but an immediate control run favored the old layer-20 default by 8.7% to 17.1%. | Five-fixture gate passed with `tensor_vs_standard` worst RMS `0.139912` and worst top20 abs `0.316128`. | Not promoted because the speed win was not stable. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=18` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=18` plus `DS4_METAL_MPP_MOE_DOWN_START_LAYER=20` | Slower than the promoted Tensor auto default by 0.1% to 3.6% in two-repeat median timing. | Not run. | Reject before drift gate. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=18` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=18` with down defaulting to 19 | Two-repeat median vs 19/19/19 Tensor auto: +0.1% at 512, then -0.7%, -1.9%, -3.0%, and -1.3% from 1024..8192. Generation was within -0.9%..+0.6%. | Not run. | Reject before drift gate because it is slower at most measured contexts. |
| `DS4_METAL_MPP_MOE_TILE_N=64` | Slower than default by 3.3% to 15.6%. | Not run. | Reject before drift gate. |
| `DS4_METAL_MPP_MOE_DOWN_START_LAYER=18` with gate/up/down defaulting to 19/19/19 | Two-repeat median vs 19/19/19 Tensor auto: -2.1% at 512, -3.1% at 1024, -3.3% at 2048, -0.7% at 4096, and +1.7% at 8192. Generation was within -1.2%..+0.4%. | Not run. | Reject before drift gate because it is slower at most measured contexts. |
| `DS4_METAL_MPP_F16_PAIR=1` | Slower than default by 0.9% to 8.6%. | Previously known safe, but not rerun here. | Keep opt-in. |
| `DS4_METAL_MPP_ATTN_OUT_TILE_N=32` | Slower than default by 1.1% to 16.4%. | Not run. | Keep default tile 64. |
| `DS4_METAL_MPP_ATTN_OUT_FILTER=layer=31..42` | Two-repeat median vs 32..42 Tensor auto: flat at 512, then slower by 0.3% to 1.4% from 1024..8192. | Not run. | Reject before drift gate; keep attention-output at 32..42. |
| `DS4_METAL_MPP_MOE_PAIR_GATE_UP=1` with gate/up/down defaulting to 19/19/19 | Two-repeat median vs 19/19/19 Tensor auto: -6.2% at 512, -3.4% at 1024, -2.7% at 2048, -2.5% at 4096, and -2.1% at 8192. Generation was within -0.2%..+1.2%. | Not run. | Reject before drift gate because the paired dispatch is consistently slower. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus `DS4_METAL_EXPERIMENTAL_MOE_MATMUL_START_LAYER=10` | Two-repeat median vs current Tensor auto: +7.5% at 512, +8.4% at 1024, +6.0% at 2048, +3.8% at 4096, +4.8% at 8192. Generation was -2.8%, -1.0%, +1.3%, +1.1%, +0.7%. | Failed the five-fixture gate: `long_memory_archive` top-1 changed and greedy differed at step 0; `tensor_vs_standard` also had one top-1 and one greedy mismatch. | Reject despite the speed because it violates the no-new-top1/no-new-greedy rule. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus `DS4_METAL_EXPERIMENTAL_MOE_MATMUL_START_LAYER=12` | Two-repeat median vs current Tensor auto: +12.2% at 512, +8.5% at 1024, +8.3% at 2048, +3.2% at 4096, +1.1% at 8192. Generation was +3.4%, -0.2%, +1.5%, -4.6%, -3.6%. | Full `./ds4_test --metal-mpp-equivalence` passed with no top-1 or greedy mismatch, but drift rose to worst RMS `0.300474` and worst top20 abs `1.00957`. | Reject before the full quality gate: long-context speed is weak and drift is much worse than the current conservative default. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus `DS4_METAL_EXPERIMENTAL_MOE_MATMUL_START_LAYER=15` | Two-repeat median vs current Tensor auto: +2.3% at 512, +2.0% at 1024, +1.5% at 2048, +2.6% at 4096, +2.0% at 8192. Generation was -2.7%, +0.0%, -1.8%, +1.1%, +1.4%. | Full `./ds4_test --metal-mpp-equivalence` passed with no top-1 or greedy mismatch, but drift rose to worst RMS `0.229322` and worst top20 abs `0.511806`. | Reject before the full quality gate: speed is marginal and drift is still worse than default. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus `DS4_METAL_EXPERIMENTAL_MOE_MATMUL_START_LAYER=17` | Two-repeat median vs current Tensor auto: +2.2% at 512, +0.5% at 1024, +0.8% at 2048, +1.2% at 4096, +0.7% at 8192. Generation was within -1.7%..+0.5%. | Full `./ds4_test --metal-mpp-equivalence` passed with no top-1 or greedy mismatch, but drift rose to worst RMS `0.190587` and worst top20 abs `0.560192`. | Reject before the full quality gate: speed is within noise and drift is worse than default. |

## Promoted Candidates

| Candidate | Speed result | Drift result | Decision |
| --- | --- | --- | --- |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=19` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=19` plus `DS4_METAL_MPP_MOE_DOWN_START_LAYER=21` | Two-repeat median vs current Tensor auto: +0.6% at 512, +0.8% at 1024, +2.3% at 2048, +2.0% at 4096, +1.6% at 8192. Generation was within -1.4%..+0.5%. | Five-fixture gate passed, first as env candidate and again as the env-free default after promotion. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1` matching standard-vs-quality, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.176030`, worst top20 abs `0.360397`. | Promoted, then superseded by the lower-drift 19/19/20 window and the faster 19/19/19 window. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=19` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=19` plus `DS4_METAL_MPP_MOE_DOWN_START_LAYER=20` | Two-repeat median vs 19/19/21 Tensor auto: +0.3% at 512, +1.2% at 1024, +0.9% at 2048, +0.4% at 4096, +0.2% at 8192. Generation was within -0.9%..+1.0%. | Five-fixture gate passed, first as env candidate and again as the env-free default after promotion. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1` matching standard-vs-quality, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.066747`, worst top20 abs `0.191437`. | Promoted, then superseded by the slightly faster 19/19/19 window. |
| `DS4_METAL_MPP_MOE_DOWN_START_LAYER=19` with gate/up unchanged at 19 | Two-repeat median vs 19/19/20 Tensor auto: +0.9% at 512, +1.2% at 1024, +1.1% at 2048, +0.4% at 4096, +0.9% at 8192. Generation was within -1.0%..+1.4%. | Five-fixture env-candidate gate passed. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1` matching standard-vs-quality, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.136143`, worst top20 abs `0.315292`. | Promoted as the next routed-MoE default window: gate/up/down from layer 19. |

## Default-Off Candidates

| Candidate | Speed result | Drift result | Decision |
| --- | --- | --- | --- |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` | Two-repeat median vs current Tensor auto: +15.9% at 512, +19.7% at 1024, +12.5% at 2048, +6.8% at 4096, +11.7% at 8192. Generation was -4.9%, -1.5%, -3.5%, -0.9%, -1.7%. | Five-fixture gate passed. `tensor_vs_quality` stayed inside the current standard-vs-quality envelope with top1 mismatches `0`, greedy mismatches `1`, worst RMS `0.618172`, and worst top20 abs `2.24006`. `tensor_vs_standard` had no top1 or greedy mismatch, but drift increased to worst RMS `0.669241` and worst top20 abs `1.30664`. | Keep default-off until an eval confirms the larger Tensor-vs-standard logit movement is acceptable. This is the best prefill candidate so far, but not yet promoted over the lower-drift conservative default. |

## Profile Signal

Representative profile:

```sh
env DS4_METAL_GRAPH_TOKEN_PROFILE=1 \
    DS4_METAL_LAYER_STAGE_PROFILE=1 \
    DS4_METAL_MOE_STAGE_PROFILE=1 \
    DS4_METAL_MOE_STAGE_PROFILE_FILTER=gate \
    DS4_METAL_ATTN_OUT_STAGE_PROFILE=1 \
    ./ds4 --metal -mt auto \
      --prompt-file tests/test-vectors/prompts/long_code_audit.txt \
      -c 8192 -n 1 --system "" --nothink --temp 0
```

Result: `prefill: 407.88 t/s`.

Important stage timings at `tokens=3844`:

- Early routed MoE before Tensor MoE window: about `99-125 ms/layer`.
- Routed MoE after gate/up Tensor starts at layer 20 in the original baseline:
  about `64 ms/layer`.
- Routed MoE after down Tensor starts at layer 22 in the original baseline:
  about `44 ms/layer`.
- Attention `q_path`: about `25 ms/layer`.
- Attention output projection: about `37 ms/layer`.

The routed-MoE stage profiler now prints layer, token/pair counts, expert
count, gate/down quant types, `mm_id` vs `mm_id_pair_mpp` path, active Tensor
route mask, tile widths, and intermediate precision. Use
`DS4_METAL_MOE_STAGE_PROFILE_FILTER=<substring>` to limit printed rows while
preserving stage flushes for timing correctness.

Long-shape routed-MoE profile on `long_code_audit`, `tok=3844`,
`pairs=23064`, `experts=6`, `gate=iq2_xxs`, `down=q2_k`:

- `FILTER=gate`: layers 0..19 use legacy `mm_id` (`mpp=0/0/0`) and gate is
  about `32-37 ms`; layers 20..42 use Tensor gate/up (`mpp=1/1/0` or
  `1/1/1`) and gate is about `13.6-14.3 ms`.
- `FILTER=down`: layers 0..21 use legacy down (`mpp=0/0/0` or `1/1/0`) and
  down is about `32-39 ms`; layers 22..42 use Tensor down (`mpp=1/1/1`) and
  down is about `13.0-13.9 ms`.

This confirms the highest-value routed-MoE target is still the pre-window
specialized `mm_id` path, not the generic dense Q8_0 wrapper. The dense
attention target remains `attn_q_b in=1024 out=32768`.

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
