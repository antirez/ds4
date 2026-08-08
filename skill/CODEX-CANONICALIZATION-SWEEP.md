# Composable canonicalization sweep

## Status

- Starting `{QA}` baseline: **PROVEN BY TEST** on M4 Max.
- KV producer localization: **PROVEN BY SOURCE**.
- KV isolated primitive A/B and `{QA,KV}` causal substitution:
  **PROVEN BY TEST** on M4 Max.
- Previous coarse `{QA,KV}` first divergence: `CP4/after_attn_hc` at row 0,
  layer 0: **PROVEN BY TEST** on M4 Max.
- Natural semantic subdivision of the interval leading to CP4:
  **PROVEN BY TEST** on M4 Max.
- Previous first divergence: `CP2-Q-CUR/q_cur` at row 0, layer 0;
  `CP2-Q-NORM/qr_norm` exact: **PROVEN BY TEST** on M4 Max.
- Q-B isolated A/B and `{QA,KV,QB}` causal substitution:
  **PROVEN BY TEST** on M4 Max.
- Current first divergence: `CP4-HEADS/attn_heads` at row 0, layer 0;
  `CP2-Q-CUR/q_cur` exact: **PROVEN BY TEST** on M4 Max.
- Pre-inverse-RoPE heads subdivision and raw-attention canonicalization:
  **PROVEN BY TEST** on M4 Max.
- Current first divergence after `{QA,KV,QB,ATTN-RAW}`:
  `CP4/after_attn_hc` at row 0, layer 0: **PROVEN BY TEST** on M4 Max.
- Isolated same-input CP4 tail A/B and conditional `CP4-TAIL`
  causal substitution: **IMPLEMENTED; M4 RUNTIME REQUIRED**.
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
DS4_FIRST_DIVERGENCE_CANONICAL=QA,KV,QB
DS4_FIRST_DIVERGENCE_CANONICAL=QA,KV,QB,ATTN-RAW
DS4_FIRST_DIVERGENCE_CANONICAL=QA,KV,QB,ATTN-RAW,CP4-TAIL
```

`KV` without `QA`, `QB` without both earlier sites, `ATTN-RAW` without
`QA,KV,QB`, or `CP4-TAIL` without the complete earlier set is rejected because
this forward sweep preserves the proven causal prefix. The legacy
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
next localization target.  That result gates creation of the `QB` mask bit
described below.

The real M4 result was:

```text
FIRST_DIVERGENCE row=0 layer=0 checkpoint=CP2-Q-CUR subobject=q_cur
ATTENTION_INTERVAL_STAGE stage=CP2-Q-NORM semantic=normalized_q_a_projection_output subobject=qr_norm result=EXACT elements=1024
ATTENTION_INTERVAL_STAGE stage=CP2-Q-CUR semantic=q_after_q_b_head_norm_and_rope subobject=q_cur result=MISMATCH elements=32768 mismatch_count=26485 mismatch_fraction=0.808258056640625 max_abs=1.9073486328125e-06 mean_abs=1.0823077339905396e-07 rms_abs=1.5083385413086959e-07 p50_abs=8.9406967163085938e-08 p95_abs=2.9802322387695312e-07 p99_abs=4.76837158203125e-07 relative_l2=1.5091385773441392e-07 max_rel=0.0027780565314971164 max_ulp=30674 cosine_similarity=0.99999999999997191 positive_delta_count=13379 negative_delta_count=13106
ATTENTION_INTERVAL_RESULT FIRST_RUNTIME_DIVERGENCE stage=CP2-Q-CUR semantic=q_after_q_b_head_norm_and_rope subobject=q_cur
```

## CP2-Q-CUR producer localization

On Apple Metal, `ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor()` is an
explicit unavailable stub that returns zero.  The actual runtime producer is
therefore the existing fallback:

| Stage | Generic verifier | Canonical sequential decode |
| --- | --- | --- |
| Q-B projection | `metal_graph_matmul_q8_0_named_tensor("attn_q_b", ..., n_tokens)` | `metal_graph_matmul_dense_quant_abs(..., n=1)` |
| Weight | same `layer->attn_q_b`, Q8_0 bytes and metadata | same |
| Input | F32 `qr_norm`, `[n_tokens, 1024]` | F32 row, `[1, 1024]` |
| Projection output | F32 Qraw, `[n_tokens, 32768]` | F32 row, `[1, 32768]` |
| Post-projection | `ds4_gpu_head_rms_norm_rope_tail_tensor(..., n_tokens)` | the same helper with `n_tok=1` |

The post-projection kernel dispatches one independent threadgroup per
token/head and uses the same reduction and RoPE implementation.  The Q-B
projection itself selects the same Q8_0 batch-ext versus single-MV primitive
pair already proven at QA and KV.  Source evidence therefore places Q-B in
`FAMILY_Q8_0_BATCH_EXT_VS_SINGLE_MV`; causal removal remains pending runtime.

When `QB` is selected, the diagnostic:

1. runs a same-input isolated Q-B projection A/B on the captured real
   `cp2_q_norm` allocation, stopping before RMS normalization and RoPE;
2. requires input bits and the Q8_0 weight object to match;
3. runs generic Q-B once per verifier batch and sequential Q-B on row 0;
4. canonicalizes Q-B for every verifier row/layer through the existing N=1
   row helper;
5. leaves the shared head RMS/RoPE kernel, attention, compressor, output
   projection, HC expansion, and sequential path unchanged.

Run:

```sh
DS4_FIRST_DIVERGENCE=1 \
DS4_FIRST_DIVERGENCE_CANONICAL=QA,KV,QB \
DS4_DSPARK_SCHEDULER=0 \
./ds4 --dspark --dspark-confidence 0 \
  -m ./ds4flash.gguf \
  --mtp ./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --tokens 16 --temp 0 --nothink \
  -p 'Explain Redis in one sentence.' \
  >canonical-sweep-qa-kv-qb.log 2>&1
```

Inspect:

```sh
grep -E '^QA_PRIMITIVE_|^KV_PRIMITIVE_|^QB_|^C2B_|^FIRST_DIVERGENCE |^ATTENTION_INTERVAL_|^CANONICALIZATION_SWEEP_|^SWEEP_STEP |^DRIFT_SOURCE' \
  canonical-sweep-qa-kv-qb.log
```

Required gates:

```text
QB_PRIMITIVE_AB input_bits_equal=PASS weights_same=PASS ... result=MISMATCH ...
QB_PRIMITIVE_NUMERICAL_NON_EQUIVALENCE PROVEN_BY_TEST
C2B_CONTROL A0_vs_A1 PASS
C2B_PROBE   A0_vs_A2 PASS
C2B_RESULT  PASS
ATTENTION_INTERVAL_STAGE stage=CP2-Q-CUR ... result=EXACT
SWEEP_STEP index=2 added=QB ... previous_stage_after_patch=EXACT ...
```

## Divergent operator families

| Site | Semantic | Generic producer | Sequential producer | Family | Evidence | Canonicalization removes site? |
| --- | --- | --- | --- | --- | --- | --- |
| CP2-Q | Q-A projection output | named Q8_0 batch projection | dense-quant N=1 projection | `FAMILY_Q8_0_BATCH_EXT_VS_SINGLE_MV` | source + isolated A/B + causal substitution | YES |
| CP2-KV-P | pre-store raw KV projection output | named Q8_0 batch projection | dense-quant N=1 projection | `FAMILY_Q8_0_BATCH_EXT_VS_SINGLE_MV` | source + isolated A/B + causal substitution | YES |
| CP2-Q-CUR | Q-B projection before shared head RMS/RoPE | named Q8_0 batch projection | dense-quant-abs N=1 projection | `FAMILY_Q8_0_BATCH_EXT_VS_SINGLE_MV` | source; isolated A/B and causal substitution pending | UNKNOWN |

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

The next source after `{QA,KV}` is **PROVEN BY TEST** at
`CP2-Q-CUR/q_cur`.  Source localization identifies the Q-B projection as the
first testable member of the compound producer; its isolated A/B and causal
removal are **UNKNOWN pending M4 runtime**.  No attention, output projection,
HC expansion, FFN, MoE, store, or later checkpoint is canonicalized before
Q-B is adjudicated.

## CP4-HEADS subdivision and raw-attention family

The `{QA,KV,QB}` M4 run established:

```text
ATTENTION_INTERVAL_STAGE stage=CP2-Q-CUR ... result=EXACT
FIRST_DIVERGENCE row=0 layer=0 checkpoint=CP4-HEADS subobject=attn_heads
```

Both paths already materialize the same F32 attention-head object before
inverse RoPE mutates it in place. The diagnostic now snapshots that object as
`CP4-HEADS-RAW/attn_heads_raw`. It does not expose or recompute logits,
softmax probabilities, or `P·V`; the batch and single-row FlashAttention
kernels do not materialize a common intermediate for those stages.

At row 0, layer 0 of the fixed Flash model, the producer pair is:

| Path | Producer | Metal arithmetic |
| --- | --- | --- |
| generic | `ds4_gpu_attention_decode_raw_batch_heads_tensor` | direct `kernel_flash_attn_ext_f16_dk512_dv512` batch output |
| sequential | `ds4_gpu_attention_decode_heads_tensor` | `kernel_flash_attn_ext_vec_f16_dk512_dv512` plus `kernel_flash_attn_reduce` |

If `CP4-HEADS-RAW` is the first mismatch, classify it as
`FAMILY_FLASH_ATTN_BATCH_DIRECT_VS_SINGLE_VEC_REDUCE`. This is a new
arithmetic family, independent of the proven Q8_0 projection family.

First run the new checkpoint without attention canonicalization:

```sh
DS4_FIRST_DIVERGENCE=1 \
DS4_FIRST_DIVERGENCE_CANONICAL=QA,KV,QB \
DS4_DSPARK_SCHEDULER=0 \
./ds4 --dspark --dspark-confidence 0 \
  -m ./ds4flash.gguf \
  --mtp ./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --tokens 16 --temp 0 --nothink \
  -p 'Explain Redis in one sentence.' \
  >canonical-sweep-attn-raw-baseline.log 2>&1
```

If `CP4-HEADS-RAW` mismatches, run the narrow diagnostic substitution:

```sh
DS4_FIRST_DIVERGENCE=1 \
DS4_FIRST_DIVERGENCE_CANONICAL=QA,KV,QB,ATTN-RAW \
DS4_DSPARK_SCHEDULER=0 \
./ds4 --dspark --dspark-confidence 0 \
  -m ./ds4flash.gguf \
  --mtp ./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --tokens 16 --temp 0 --nothink \
  -p 'Explain Redis in one sentence.' \
  >canonical-sweep-attn-raw-canonical.log 2>&1
```

`ATTN-RAW` only disables the raw-only batched attention call and reuses the
existing per-row `ds4_gpu_attention_decode_heads_tensor` fallback. Compressed
attention, inverse RoPE, output projection/fusion, replay, and production
behavior remain unchanged.

## Isolated CP4 tail A/B

No additional first-divergence checkpoint is introduced between `CP4-HEADS`
and CP4. The existing generic CP4 capture also snapshots the tail's F32
`cur_hc` and `hc_split` inputs into diagnostic-owned tensors. These snapshots
are A/B fixtures, not ordered comparison objects.

The isolated layer-0 test clones the exact same `CP4-HEADS`, `cur_hc`, and
`hc_split` bits for both calls and uses the same model mapping, Q8_0 output-A
weights, and Q8_0 output-B weights:

| Path | Exact primitives |
| --- | --- |
| generic | `metal_graph_attention_output_dense_quant_batch` then `ds4_gpu_hc_expand_split_tensor` |
| sequential | per-row `ds4_gpu_attention_output_low_q8_tensor` then `ds4_gpu_matmul_q8_0_hc_expand_tensor` |

On Metal, the batch-F16 output and half-HC helpers are unavailable stubs.  The
ordinary generic verifier therefore uses this F32 fallback.  The isolated test
must use the same fallback instead of treating the unavailable fast-path probe
as a hard failure.  The test covers output-A, output-B, HC expansion, and
residual addition. Because the sequential path fuses output-B with HC expansion,
attribution to one component inside the compound tail remains `UNKNOWN`. Only
the final F32 `after_attn_hc` is compared bitwise.

Run the isolated A/B without tail substitution:

```sh
DS4_FIRST_DIVERGENCE=1 \
DS4_FIRST_DIVERGENCE_CANONICAL=QA,KV,QB,ATTN-RAW \
DS4_DSPARK_SCHEDULER=0 \
./ds4 --dspark --dspark-confidence 0 \
  -m ./ds4flash.gguf \
  --mtp ./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --tokens 16 --temp 0 --nothink \
  -p 'Explain Redis in one sentence.' \
  >canonical-sweep-cp4-tail-ab.log 2>&1
```

Required semantic gate:

```text
CP4_TAIL_PRIMITIVE_AB input_bits_equal=PASS weights_same=PASS semantics_preserved=PASS ... result=MISMATCH|EXACT ... evidence=PROVEN_BY_TEST
```

If it is `MISMATCH`, the compound candidate family is proven by the isolated
test. `CP4-TAIL` first repeats the run with the earlier four canonicalizations,
runs the isolated A/B, and enables row-wise sequential tail substitution only
after the A/B proves numerical non-equivalence in that same invocation:

```sh
DS4_FIRST_DIVERGENCE=1 \
DS4_FIRST_DIVERGENCE_CANONICAL=QA,KV,QB,ATTN-RAW,CP4-TAIL \
DS4_DSPARK_SCHEDULER=0 \
./ds4 --dspark --dspark-confidence 0 \
  -m ./ds4flash.gguf \
  --mtp ./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --tokens 16 --temp 0 --nothink \
  -p 'Explain Redis in one sentence.' \
  >canonical-sweep-cp4-tail-canonical.log 2>&1
```

The substitution is diagnostic-only. It creates no new online checkpoint,
does not alter sequential Pass B, and leaves production execution and replay
unchanged when the mask is unset.
