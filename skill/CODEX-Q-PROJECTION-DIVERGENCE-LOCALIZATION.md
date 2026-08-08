# Q projection divergence localization

## Scope and conclusion

The M4 result fixes the first mismatch at row 0, layer 0, CP2-Q/`qr`, while
CP1/`attn_norm` immediately before it is bit-exact.  In both production paths,
`qr` is the direct F32 output of the first Q-A projection.  There is no
naturally materialized semantic object between CP1 and `qr`, so the frozen
stop rule applies: the first Q projection already differs and localization
must not enter kernel internals.

No GPU checkpoint, dispatch, synchronization, allocation, or execution-order
change is introduced.  The specialized report reuses the existing C2b/C4/C5
captures and `ds4_float_compare_exact()`.

## Narrow Q-side source map

| Stage | Generic producer/object | Sequential producer/object | Same mathematical object? | Naturally materialized both sides? | Comparable without path change? |
| --- | --- | --- | --- | --- | --- |
| CP1 normalized attention input | attention norm / `metal_graph_batch_attn_norm(g)` | attention norm / `metal_graph_attn_norm(g)` | Yes | Yes | Yes; existing CP1 |
| First Q-A projection output | `metal_graph_matmul_q8_0_named_tensor("attn_q_a", ...)` / `metal_graph_batch_qr(g)` | `metal_graph_matmul_dense_quant_tensor(... layer->attn_q_a ...)` / `metal_graph_qr(g)` | Yes | Yes | Yes; existing CP2-Q |
| Q-A normalized output | Q RMS norm / `metal_graph_batch_qr_norm(g)` | Q RMS norm / `metal_graph_qr_norm(g)` | Yes | Yes | Yes, but after CP2-Q and outside the frozen interval |
| Q-B raw output | generic fused Q-B/head-normalization/RoPE route | sequential Q-B route | Conceptually yes | No on the generic real path | No; exposing it would change fusion selection |
| Per-head normalized Q | generic fused Q-B/head-normalization/RoPE route | sequential head-normalization route | Conceptually yes | Not reliably | No; exposing it would change fusion selection |
| Post-RoPE Q | final `metal_graph_batch_q(g)` | final `metal_graph_q(g)` | Yes | Yes | Downstream of CP2-Q; outside this task |

Detailed storage properties for the only two boundaries in the interval:

| Semantic object | Dtype and shape | Row layout | First valid through reuse | Available |
| --- | --- | --- | --- | --- |
| normalized attention input | F32 `[DS4_N_EMBD]` | contiguous; stride `DS4_N_EMBD * sizeof(float)` | after attention norm; until layer scratch reuse | Yes, existing CP1 |
| Q-A projection output | F32 `[q_rank]` (`1024` elements in the observed model) | contiguous; stride `q_rank * sizeof(float)` | after `attn_q_a` projection; `qr_norm` is separate, so until layer scratch reuse | Yes, existing CP2-Q |

The exact chain up to the frozen outer boundary is:

```text
generic verifier
metal_graph_batch_attn_norm(g)
  -> metal_graph_matmul_q8_0_named_tensor("attn_q_a", ...)
  -> metal_graph_batch_qr(g)                 [existing CP2-Q]

canonical sequential target (Apple single-device)
metal_graph_attn_norm(g)
  -> metal_graph_matmul_dense_quant_tensor(... layer->attn_q_a ...)
  -> metal_graph_qr(g)                       [existing CP2-Q]
```

On this fixed Apple single-device platform `g->cuda_qkv_pair` is false, so the
CUDA-only joint Q/KV projection branch is not the Pass B producer.  After the
existing CP2-Q boundary, both paths consume `qr` to produce `qr_norm`, then
perform Q-B projection, per-head normalization, and RoPE.  Those later
operators are outside the already-proven CP1-to-CP2-Q interval.

For row `r`, generic snapshot storage is contiguous row-major with byte offset
`r * width * sizeof(float)`.  Sequential Pass B captures the single-token
tensor into that same row offset.  Row 0 therefore maps to proposal token 0 in
Pass A and to the explicitly forced `forced_tokens[0]` invocation in Pass B.

CP1 becomes valid after the attention normalization producer.  CP2-Q becomes
valid immediately after the Q-A projection and remains live while the distinct
`qr_norm` tensor is produced.  Existing same-encoder inline copies preserve
both values before layer scratch is reused.

## Report contract

When the ordinary report establishes the first divergence as row 0, layer 0,
CP2-Q/`qr`, it also emits:

```text
Q_DIVERGENCE_TRACE row=0 layer=0
Q_DIVERGENCE_STAGE stage=CP1 semantic=normalized_attention_input subobject=attn_norm result=EXACT ...
Q_DIVERGENCE_STAGE stage=CP2-Q semantic=q_a_projection_output subobject=qr result=MISMATCH ...
Q_FIRST_DIVERGENCE stage=q_a_projection_output ...
Q_LOCALIZATION_RESULT FIRST_RUNTIME_DIVERGENCE_WITHIN_Q_PATH
```

`Q_FIRST_DIVERGENCE` includes element count, mismatch count, first mismatch
index, actual and expected bits, maximum absolute and relative differences,
and maximum ULP distance.  “Actual” is generic Pass A; “expected” is canonical
sequential Pass B.

The trace is a hard sanity gate.  Missing or incompatible snapshots, a CP1
mismatch, or an exact CP2-Q result emits `Q_DIVERGENCE_SANITY FAIL` and makes
the report fail instead of claiming a localization.

No micro-checkpoint was added: the existing CP2-Q is already the first Q-A
projection output, and there is no intervening natural tensor to capture.
The exact capture locations remain the existing post-layer same-encoder inline
copies in `ds4_c2b_capture_attention()` for generic Pass A and
`ds4_c45_capture_layer()` for sequential Pass B.  Both copy the still-live F32
producer outputs with `ds4_gpu_tensor_copy_f32_inline()`; neither introduces a
readback or command-buffer completion inside the layer loop.

## M4 Max command

```sh
DS4_FIRST_DIVERGENCE=1 \
DS4_DSPARK_SCHEDULER=0 \
./ds4 --dspark --dspark-confidence 0 \
  -m ./ds4flash.gguf \
  --mtp ./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --tokens 16 --temp 0 --nothink \
  -p 'Explain Redis in one sentence.' \
  >q-path-localization.log 2>&1
```

Inspect the machine-readable result with:

```sh
grep -E '^C2B_|^FIRST_DIVERGENCE |^Q_' q-path-localization.log
```

The implementation environment is not an M4 Max, so it must not claim a
runtime localization result.  The command above is the required M4 rerun.
