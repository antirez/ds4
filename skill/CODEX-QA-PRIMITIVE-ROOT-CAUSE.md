# QA primitive numerical experiment

## Status

- Phase A isolated primitive A/B: **IMPLEMENTED; M4 RUNTIME REQUIRED**.
- Phase B QA-only canonicalization: **BLOCKED** until Phase A prints
  `QA_PRIMITIVE_NUMERICAL_NON_EQUIVALENCE PROVEN_BY_TEST` on the M4 run.
- Production behavior is unchanged when `DS4_QA_PRIMITIVE_AB` is unset.

## Proven runtime facts

The preceding M4 whole-model experiment established:

```text
row=0 layer=0
CP1/attn_norm: EXACT, 4096 F32 elements
CP2-Q/qr: MISMATCH, 823/1024 elements
first bits: generic=0x3b21c760 sequential=0x3b21c7a0
max_abs=5.2154064178466797e-08
```

Evidence label: **PROVEN BY TEST** for the whole-model checkpoint boundary.
It does not by itself prove primitive-level non-equivalence in isolation.

## Exact producer pair

Both high-level helpers ultimately call `ds4_gpu_matmul_quant_tensor()` with
the same Q8_0 weight tensor.  Their runtime row count selects different Metal
dispatches.

| Property | Generic verifier | Canonical sequential decode |
| --- | --- | --- |
| Producer | `metal_graph_matmul_q8_0_named_tensor("attn_q_a", ...)` | `metal_graph_matmul_dense_quant_tensor(... layer->attn_q_a ...)` |
| Input | `metal_graph_batch_attn_norm(g)` | `metal_graph_attn_norm(g)` |
| Weight | `layer->attn_q_a`, same model map and `abs_offset` | same |
| Output | `metal_graph_batch_qr(g)` | `metal_graph_qr(g)` |
| Input/output dtype | F32 / F32 | F32 / F32 |
| Weight representation | Q8_0: one half scale plus 32 signed int8 values, 34 bytes per block | same bytes |
| M/N/K | M=1024, N=proposal rows (2–16), K=4096 | M=1024, N=1, K=4096 |
| Input row stride | `4096 * sizeof(float)` | single contiguous row |
| Output row stride | `1024 * sizeof(float)` | single contiguous row |
| Default Metal dispatch | `kernel_mul_mv_ext_q8_0_f32_r1_*` | `kernel_mul_mv_q8_0_f32` |
| Accumulator/store | F32 accumulator and F32 store | F32 accumulator and F32 store |

Evidence label: **PROVEN BY SOURCE**.

## Source-proven implementation differences

With the diagnostic's real 4096-wide input and the default small-batch
threshold, generic N=2–16 enters the `mul_mv_ext` branch.  It dequantizes Q8_0
chunks to `float4`, accumulates `dot(float4, float4)`, and reduces with
`simd_shuffle_down` over the lanes assigned to one output row.

Sequential N=1 enters the classic Q8_0 matvec.  Each lane explicitly sums
eight `int8 * float` products, multiplies each block subtotal by the Q8_0
scale, accumulates in F32, then performs a SIMD reduction followed by a
threadgroup reduction across the default four simdgroups on a single device.

The source therefore proves different dequantization expression, work
partition, and reduction tree.  It does not prove the hardware's exact FMA
contraction behavior.  FMA contraction is **UNKNOWN** and is not used as an
explanation before the isolated runtime result.

## Phase A real-shape isolated A/B

`DS4_QA_PRIMITIVE_AB=1` reuses the layer-0 CP1 GPU snapshot produced by the
existing real Pass A.  The generic primitive consumes all original verifier
rows.  The sequential primitive consumes a row-0 view at byte offset zero of
that exact same allocation.  No transpose, repack, recomputation, or fixture
generation occurs.

Both invocations use the same `ds4_model`, `layer->attn_q_a` pointer, Q8_0
metadata, model mapping, weight offset, M=1024, and K=4096.  Only N/dispatch is
allowed to differ.  Dedicated output buffers prevent aliasing.  CPU comparison
occurs after GPU completion through `ds4_float_compare_exact()`.

Run on M4 Max:

```sh
DS4_FIRST_DIVERGENCE=1 \
DS4_QA_PRIMITIVE_AB=1 \
DS4_DSPARK_SCHEDULER=0 \
./ds4 --dspark --dspark-confidence 0 \
  -m ./ds4flash.gguf \
  --mtp ./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --tokens 16 --temp 0 --nothink \
  -p 'Explain Redis in one sentence.' \
  >qa-primitive-ab.log 2>&1
```

Inspect:

```sh
grep -E '^C2B_|^QA_PRIMITIVE_|^FIRST_DIVERGENCE ' qa-primitive-ab.log
```

Required proof line:

```text
QA_PRIMITIVE_AB input_bits_equal=PASS weights_same=PASS elements=1024 result=MISMATCH ...
QA_PRIMITIVE_NUMERICAL_NON_EQUIVALENCE PROVEN_BY_TEST
```

If output is exact, setup fails, or the real verifier contains only one row,
the diagnostic prints `QA_PRIMITIVE_ISOLATION_INCONCLUSIVE`, skips subsequent
Pass B/report work, and exits nonzero.

Exact reproduction of `0x3b21c760` versus `0x3b21c7a0` is desirable but is not
required by the gate.  The mandatory evidence is same real input and weight
bytes producing a non-bit-identical 1024-element output.

## Phase A result

Current status: **M4 RUNTIME PENDING**.

The implementation host cannot run the Apple Metal model.  No isolated A/B
result is claimed by this commit.

## Phase B gate

QA-only canonicalization remains unimplemented until the M4 log proves Phase
A.  After that proof, a second focused commit may add a default-off
`DS4_FIRST_DIVERGENCE_CANONICAL_QA=1` mode which runs only generic Q-A once per
proposal row through the canonical N=1 primitive and leaves the rest of the
generic verifier batched and unchanged.

Old first divergence:

```text
row=0 layer=0 checkpoint=CP2-Q subobject=qr
```

New first divergence: **UNKNOWN pending Phase B**.
