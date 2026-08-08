# Codex C4/C5 Implementation Note

## Gate and scope

- Parent commit: `7f68d7f2f8eafe3b7dbb4a3f08829ce27fd55b07`.
- C2b was reported PASS on Apple M4 Max for that commit.
- This commit adds only the canonical forced-token Pass B and the C5 CPU comparison/report.
- Replay, proposal generation, acceptance, scheduler behavior, kernels, and fusion selection remain unchanged.

## Canonical Pass B

After the existing A0/A1/A2 gate passes, the diagnostic restores the original
S0 again with `spec_frontier_restore()`, the pre-A0 raw-KV backups, and
`checkpoint.len = start`. It then forces the immutable proposal vector through:

```text
for row = 0..proposal_count-1
    metal_graph_eval_token_raw_swa(... proposal_tokens[row], start + row ...)
    token_vec_push(... proposal_tokens[row])
```

This is the same ordinary sequential target decode and push ordering used by
normal DSpark replay. Pass B never selects or replaces a token with argmax.

## Sequential checkpoint hook

The hook runs immediately after `metal_graph_encode_decode_layer()` returns and
before `cur_hc` / `after_ffn_hc` pointer swap. At that boundary:

| Checkpoint | Sequential source | Validity at hook |
|---|---|---|
| CP1 | `metal_graph_attn_norm(g)` | normalized attention input remains live |
| CP2-Q | `metal_graph_qr(g)` | Q-A projection output remains live and F32 |
| CP2-KV-P | `metal_graph_kv_raw(g)` | raw KV projection output remains live and F32 |
| CP2-KV-R | `layer_raw_cache[il][pos % raw_cap]` | ordinary one-row store is already encoded |
| CP3-P | — | omitted; no natural sequential semantic-equivalent object |
| CP3-F | live attention/index state tensors, counters, and emitted rows | ordinary compressor/indexer update is already encoded |
| CP4 | `metal_graph_after_attn_hc(g)` | post-attention HC remains live through FFN |
| CP5 | `metal_graph_after_ffn_hc(g)` | complete layer output before pointer swap |

Eligible F32 payloads are copied to dedicated GPU snapshot buffers with
`ds4_gpu_tensor_copy_f32_inline()` in the active compute encoder. Metal F16
attention-cache append rows remain persistent and are read in native 16-bit
storage after Pass B. No CPU tensor readback or synchronization is added inside
the layer loop. Decode graph/fusion/kernel selection is not modified.

## C5 report

After Pass B completes, its GPU snapshots are materialized into a
`ds4_first_divergence_capture`. The existing reporter compares Pass A against
canonical Pass B in frozen semantic order and emits one line per object plus:

```text
FIRST_DIVERGENCE row=<r> layer=<l> checkpoint=<cp> subobject=<name> compared_objects=<n>
```

If all captured objects are exact, it emits:

```text
FIRST_DIVERGENCE NONE compared_objects=<n>
```

A numerical mismatch is a successful C5 measurement and therefore does not
make the diagnostic process fail. Setup, restoration, Pass-B execution,
materialization, or report-generation failures exit nonzero.

## M4 Max command

Build the commit, then run:

```sh
DS4_FIRST_DIVERGENCE=1 \
DS4_DSPARK_SCHEDULER=0 \
./ds4 --dspark --dspark-confidence 0 \
  -m ./ds4flash.gguf \
  --mtp ./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --tokens 16 --temp 0 --nothink \
  -p 'Explain Redis in one sentence.' \
  >c5-first-divergence.log 2>&1
```

The run must first reproduce all three C2b PASS lines. The authoritative C5
result is the final `FIRST_DIVERGENCE ...` line; preserve the preceding object
lines as the raw comparison log.

## Local validation

```sh
make -B ds4.o first_divergence_capture.o ds4_float_compare.o
make -B ds4_cpu.o
make test-float-compare
make test-first-divergence
git diff --check
```

The implementation host is not an M4 Max. No C5 numerical result is claimed by
this commit.
