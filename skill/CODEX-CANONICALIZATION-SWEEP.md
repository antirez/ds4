# Composable canonicalization sweep

## Status

- Starting `{QA}` baseline: **PROVEN BY TEST** on M4 Max.
- KV producer localization: **PROVEN BY SOURCE**.
- KV isolated primitive A/B and `{QA,KV}` causal substitution:
  **PROVEN BY TEST** on M4 Max.
- Current `{QA,KV}` first divergence: `CP4/after_attn_hc` at row 0,
  layer 0: **PROVEN BY TEST** on M4 Max.
- Natural semantic subdivision of the interval leading to CP4:
  **IMPLEMENTED; M4 RUNTIME REQUIRED**.
- Default production behavior remains unchanged when
  `DS4_FIRST_DIVERGENCE_CANONICAL` is unset.

## Starting proven facts

The real M4 sequence established:

```text
QA_PRIMITIVE_NUMERICAL_NON_EQUIVALENCE PROVEN_BY_TEST
C2B_CONTROL A0_vs_A1 PASS
C2B_PROBE   A0_vs_A2 PASS
C2B_RESULT  PASS
CP1 EXACT
CP2-Q EXACT after canonical QA
FIRST_DIVERGENCE row=0 layer=0 checkpoint=CP2-KV-P subobject=kv_raw
```

Formal starting adjudication:

```text
QA_CAUSAL_SOURCE PROVEN
QA_ONLY_SOURCE DISPROVEN
NEXT_INDEPENDENT_DRIFT_SOURCE CP2-KV-P/kv_raw
KV_PRIMITIVE_ROOT_CAUSE NOT_YET_PROVEN
```

## Runtime canonicalization mask

The composable diagnostic interface is:

```sh
DS4_FIRST_DIVERGENCE_CANONICAL=QA
DS4_FIRST_DIVERGENCE_CANONICAL=QA,KV
```

`KV` without `QA` is rejected because this forward sweep starts from the
already-adjudicated QA correction.  The legacy
`DS4_FIRST_DIVERGENCE_CANONICAL_QA=1` gate remains accepted.  The mask is set
only while encoding generic diagnostic A0/A1/A2 and is cleared before
canonical sequential Pass B.

Every selected projection is invoked once per verifier row through the
canonical N=1 dense-quant helper.  All later generic operations consume the
row results in the existing batch tensors.  The proposal block, layer-major
schedule, Q-B, attention, compressor, FFN/MoE, cache/state handling, and
ordinary `metal_graph_eval_token_raw_swa()` reference path are unchanged.

Evidence label: **PROVEN BY SOURCE**.

## CP2-KV-P producer pair

`CP2-KV-P/kv_raw` is captured before KV RMS normalization, RoPE, FP8 cache
quantization, and persistent raw-cache storage.  It is therefore a natural
semantic comparison of the same pre-store projection output.

| Property | Generic verifier | Canonical sequential decode |
| --- | --- | --- |
| Producer | `metal_graph_matmul_q8_0_named_tensor("attn_kv", ...)` | `metal_graph_matmul_dense_quant_tensor(... layer->attn_kv ..., 1)` |
| Input semantic object | normalized attention input | same |
| Input storage | contiguous F32 rows, `[n_tokens, DS4_N_EMBD]` | contiguous F32 row, `[1, DS4_N_EMBD]` |
| Weight | the same `layer->attn_kv` object, model map, `abs_offset`, metadata, and bytes | same |
| Weight representation | Q8_0 | Q8_0 |
| Output semantic object | pre-store `kv_raw` | same |
| Output storage | contiguous F32 rows, `[n_tokens, DS4_N_HEAD_DIM]` | contiguous F32 row, `[1, DS4_N_HEAD_DIM]` |
| Accumulator/output | F32 / F32 | F32 / F32 |
| Runtime row count | proposal rows, normally 2–16 | 1 |
| Metal kernel family | `kernel_mul_mv_ext_q8_0_f32_r1_*` | `kernel_mul_mv_q8_0_f32` |

Both high-level helpers reach `ds4_gpu_matmul_quant_tensor()`.  The runtime
row count selects different Q8_0 implementations.  The generic small-batch
path dequantizes chunks to `float4`, uses `dot(float4,float4)`, and reduces
across the lanes assigned to an output row.  The single-row path explicitly
sums the Q8_0 block products, applies the block scale, and uses its SIMD plus
threadgroup reduction structure.  This proves a shared implementation family
with QA and proves different dequantization expressions, work partition, and
reduction trees.  Exact hardware FMA contraction remains **UNKNOWN**.

Evidence label: **PROVEN BY SOURCE**.

## KV isolated primitive A/B

When `KV` is selected, the diagnostic reuses layer 0's real Pass-A CP1 GPU
snapshot.  The generic KV primitive consumes the original batch allocation;
the sequential primitive consumes a row-0 view of that same allocation.
Both calls use the same `layer->attn_kv` pointer and physical Q8_0 bytes.

After GPU completion, `ds4_float_compare_exact()` performs the strict
comparison.  The additional CPU-only signature reports:

```text
elements mismatch_count mismatch_fraction
max_abs mean_abs rms_abs p50_abs p95_abs p99_abs
relative_l2 max_rel max_ulp cosine_similarity
positive_delta_count negative_delta_count
```

Absolute percentiles use nearest-rank selection.  These values characterize
the arithmetic family; they never weaken the bit-exact gate.

Required proof before KV canonicalization proceeds:

```text
KV_PRIMITIVE_AB input_bits_equal=PASS weights_same=PASS ... result=MISMATCH ...
KV_PRIMITIVE_NUMERICAL_NON_EQUIVALENCE PROVEN_BY_TEST
```

The real M4 run produced:

```text
KV_PRIMITIVE_AB input_bits_equal=PASS weights_same=PASS elements=512 result=MISMATCH mismatch_count=431 mismatch_fraction=0.841796875 max_abs=4.4703483581542969e-08 mean_abs=9.7720658231992275e-09 rms_abs=1.2926741589862414e-08 p50_abs=7.4505805969238281e-09 p95_abs=2.9802322387695312e-08 p99_abs=3.3527612686157227e-08 relative_l2=1.5454367931804077e-07 max_rel=0.00066137566137566134 max_ulp=8192 cosine_similarity=0.99999999999998945 positive_delta_count=220 negative_delta_count=211
KV_PRIMITIVE_NUMERICAL_NON_EQUIVALENCE PROVEN_BY_TEST
```

Current evidence label: **PROVEN BY TEST**.

## Sweep execution

Use the exact same target/support models, prompt, scheduler setting, and
proposal configuration for both commands.

First reproduce the frozen `{QA}` baseline:

```sh
DS4_FIRST_DIVERGENCE=1 \
DS4_FIRST_DIVERGENCE_CANONICAL=QA \
DS4_DSPARK_SCHEDULER=0 \
./ds4 --dspark --dspark-confidence 0 \
  -m ./ds4flash.gguf \
  --mtp ./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --tokens 16 --temp 0 --nothink \
  -p 'Explain Redis in one sentence.' \
  >canonical-sweep-qa.log 2>&1
```

The hard baseline gate is:

```text
CANONICALIZATION_SWEEP_BASELINE canonical_set=QA result=PASS expected_first_divergence=CP2-KV-P/kv_raw
```

Then run the first sweep step without rebuilding:

```sh
DS4_FIRST_DIVERGENCE=1 \
DS4_FIRST_DIVERGENCE_CANONICAL=QA,KV \
DS4_DSPARK_SCHEDULER=0 \
./ds4 --dspark --dspark-confidence 0 \
  -m ./ds4flash.gguf \
  --mtp ./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --tokens 16 --temp 0 --nothink \
  -p 'Explain Redis in one sentence.' \
  >canonical-sweep-qa-kv.log 2>&1
```

Inspect either log with:

```sh
grep -E '^QA_PRIMITIVE_|^KV_PRIMITIVE_|^C2B_|^FIRST_DIVERGENCE |^Q_|^KV_|^CANONICALIZATION_SWEEP_|^SWEEP_STEP |^DRIFT_SOURCE' \
  canonical-sweep-qa-kv.log
```

The `{QA,KV}` run must show CP1, CP2-Q, and CP2-KV-P exact before its new
first-divergence location is interpreted.  It then prints one `SWEEP_STEP`, a
grep-friendly `DRIFT_SOURCE_TABLE`, and `CANONICALIZATION_SWEEP_RESULT`.

The completed M4 step showed `CP2-KV-P` exact and moved the first divergence
to `CP4/after_attn_hc`.  It did not expose a new operator by itself because
CP4 spans several arithmetic stages.

## CP4 interval subdivision

The diagnostic now captures three additional existing F32 semantic objects.
No intermediate is recomputed, and the canonical sequential execution is not
changed.

| Ordered checkpoint | Semantic object | Generic tensor | Sequential tensor |
| --- | --- | --- | --- |
| `CP2-Q-NORM` | normalized Q-A projection output | `metal_graph_batch_qr_norm(g)` | `metal_graph_qr_norm(g)` |
| `CP2-Q-CUR` | Q after Q-B, per-head RMS norm, and RoPE | `metal_graph_batch_q(g)` | `metal_graph_q(g)` |
| `CP4-HEADS` | attention heads after inverse RoPE | `metal_graph_batch_heads(g)` | `metal_graph_heads(g)` |
| `CP4` | post-attention hidden state | `metal_graph_batch_after_attn_hc(g)` | `metal_graph_after_attn_hc(g)` |

The ordering follows producer causality: Q-A and KV raw projections precede
Q-A normalization and Q-B/Q-RoPE; persistent KV store and compressor frontier
precede the attention-head result; the HC post expansion ends at CP4.

The output-projection temporary is deliberately not made a frozen comparison
object.  On the tested single-device Metal configuration, generic execution
can fuse both Q8_0 output projections to F16 before HC expansion, while the
canonical sequential tail uses its existing low-rank/output/HC fusion.  A
standalone common F32 `attn_out` would therefore require changing or
recomputing a canonical path.  `CP4-HEADS` and CP4 bracket that asymmetric
fused tail without disabling fusion.

All new snapshots are preallocated and copied by
`ds4_gpu_tensor_copy_f32_inline()` from the existing end-of-layer capture
hook.  CPU comparison remains `ds4_float_compare_exact()`.  Because checkpoint
coverage changed, the M4 run must re-establish the C2b A0/A1 and A0/A2 gates.

Run the updated `{QA,KV}` diagnostic:

```sh
DS4_FIRST_DIVERGENCE=1 \
DS4_FIRST_DIVERGENCE_CANONICAL=QA,KV \
DS4_DSPARK_SCHEDULER=0 \
./ds4 --dspark --dspark-confidence 0 \
  -m ./ds4flash.gguf \
  --mtp ./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --tokens 16 --temp 0 --nothink \
  -p 'Explain Redis in one sentence.' \
  >cp4-interval-localization.log 2>&1
```

Inspect it with:

```sh
grep -E '^QA_PRIMITIVE_|^KV_PRIMITIVE_|^C2B_|^FIRST_DIVERGENCE |^ATTENTION_INTERVAL_|^CANONICALIZATION_SWEEP_|^SWEEP_STEP |^DRIFT_SOURCE' \
  cp4-interval-localization.log
```

The hard gate remains:

```text
C2B_CONTROL A0_vs_A1 PASS
C2B_PROBE   A0_vs_A2 PASS
C2B_RESULT  PASS
```

Only the first mismatching stage in `ATTENTION_INTERVAL_TRACE` becomes the
next localization target.  No new canonicalization mask bit has been added.

## Divergent operator families

| Site | Semantic | Generic producer | Sequential producer | Family | Evidence | Canonicalization removes site? |
| --- | --- | --- | --- | --- | --- | --- |
| CP2-Q | Q-A projection output | named Q8_0 batch projection | dense-quant N=1 projection | `FAMILY_Q8_0_BATCH_EXT_VS_SINGLE_MV` | source + isolated A/B + causal substitution | YES |
| CP2-KV-P | pre-store raw KV projection output | named Q8_0 batch projection | dense-quant N=1 projection | `FAMILY_Q8_0_BATCH_EXT_VS_SINGLE_MV` | source + isolated A/B + causal substitution | YES |

QA and KV satisfy the family-identity rule through the same underlying Metal
kernel pair, Q8_0 dequantization paths, batch-versus-single distinction, and
reduction mechanisms.  The M4 `{QA,KV}` run made CP2-KV-P exact, so the first
two independent sites are concentrated in one arithmetic family and the
intermediate classification is:

```text
DRIFT_FAMILY_CONCENTRATION HIGH
```

This classification covers the two established sites only.  It says nothing
about the still-unseen next source.

## Sufficient-set status

```text
SUFFICIENT_CANONICALIZATION_SET UNKNOWN
MINIMAL_CANONICAL_REPAIR_SET NOT_EVALUATED
```

The next coarse location after `{QA,KV}` is **PROVEN BY TEST** as
`CP4/after_attn_hc`.  The exact producer inside that interval remains
**UNKNOWN pending M4 runtime** with the new subdivisions.  No attention,
normalization, output projection, HC expansion, FFN, MoE, store, or later
checkpoint is canonicalized before the earliest natural semantic object is
identified.
