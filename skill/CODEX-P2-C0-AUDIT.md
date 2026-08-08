# Codex P2 C0 Static Re-audit

## Verdict

**GO** for implementation of the corrected diagnostic differential probe.

This GO means only that S0 restoration and semantic checkpoint comparison can be made credible with the corrections below. It does not authorize replay removal, direct commitment of generic verifier state, `DS4_DSPARK_EXACT_NOREPLAY`, acceptance changes, or any production optimization.

No production code was changed during C0.

## A. Source baseline

| Item | Value |
|---|---|
| Repository | `sijiaguo777/ds4` |
| Source branch | `dspark/p2-audit` |
| Working branch | `codex/p2-first-divergence` |
| Audited commit | `37b029bb884e07b32af24db0998ffd8e7fcacd45` |
| Branch relationship at audit time | identical; ahead 0, behind 0 |
| Qwen evidence commit | `72beb3b8afb6675ec363fcf55167775ccf39da89` |
| Replay correctness commit | `7fb28303df65f9eb2f22de7d5d9b6ccbda504eb5` |

Commit `72beb3b` was read as audit evidence only. `backup/qwen-p2-draft` was not checked out, merged, or cherry-picked.

## B. Corrections to the Qwen audit

| Qwen claim | Re-audit result | Evidence |
|---|---|---|
| The dual-pass direction is correct | Confirmed. Ordinary sequential target decode is the authoritative reference. | PROVEN BY SOURCE |
| `spec_frontier_restore()` fully restores S0 | Corrected. It does not restore target raw KV, append-only cache contents, HC scratch, or capture scratch. | PROVEN BY SOURCE |
| Dirty speculative raw KV is automatically benign | Rejected as a general proof. `raw_cap` can be reduced to `raw_window`, so future writes cannot universally be proven not to alias still-needed historical physical rows. | PROVEN BY SOURCE |
| Raw KV may be snapshotted after Pass A | Incorrect. The original physical rows must be captured before Pass A. | PROVEN BY SOURCE |
| Generic verification does not modify HC state | Corrected. It overwrites batch HC scratch and DSpark target-hidden capture, but no canonical cross-token persistent HC state exists in those tensors. | PROVEN BY SOURCE |
| `batch_cur_hc` can be used as the Q/KV checkpoint | Incorrect. It is the batched HC hidden workspace, not a Q/KV projection result. | PROVEN BY SOURCE |
| A batched GEMM literally accumulates across rows | Unsupported. Different kernel, tiling, reduction, fusion, and conversion paths are the source-level possibilities. | UNKNOWN |
| Compressor projection is the first divergence | Still only a hypothesis. | UNKNOWN |
| The existing layer capture hook can capture every checkpoint | Incorrect. It can support CP5-like layer-end data, but CP1-CP4 temporaries require capture at their producer boundaries. | PROVEN BY SOURCE |
| Generic and ordinary decode use different compressor paths | Confirmed at the call-path level. Current C0 did not reproduce the numerical divergence at runtime. | INFERRED |
| An exact-row wrapper is non-exact merely because it aliases another primitive | Incorrect. Aliasing alone proves neither exactness nor non-exactness. | PROVEN BY SOURCE |
| Exact-N2 disagreement invalidates the main differential experiment | Incorrect. It invalidates that auxiliary oracle until investigated; ordinary sequential remains the valid reference. | PROVEN BY SOURCE |

### Metal exact-row primitives

- `ds4_gpu_matmul_q8_0_decode_rows_exact_tensor()` has a dedicated Metal row-grid dispatch. Its exactness is **INFERRED**, because no direct bit-exact test was found in C0.
- `ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor()` aliases the generic pair primitive in the Apple stub: **UNKNOWN**.
- `ds4_gpu_matmul_f16_router_rows_exact_tensor()` aliases generic F16 matmul in the Apple stub: **UNKNOWN**.
- `ds4_gpu_attention_output_low_q8_rows_exact_tensor()` is not part of the current single-device M4 exact-N2 path: **N/A for this oracle**.

The existing `metal_graph_verify_decode2_exact()` auxiliary path calls the ordinary one-row decode layer twice per layer. It does not depend on the generic exact-row wrappers above for the M4 single-device experiment.

## C. S0 restoration proof

| State | Pass-A effect | Current restoration | Conclusion |
|---|---|---|---|
| `layer_attn_state_kv/score` | modified | full tensor copy from `spec_attn_state_*` | PROVEN BY SOURCE |
| `layer_index_state_kv/score` | modified on ratio-4 layers | full tensor copy from `spec_index_state_*` | PROVEN BY SOURCE |
| `layer_n_comp` / `layer_n_index_comp` | modified | restored from CPU snapshot arrays | PROVEN BY SOURCE |
| target `layer_raw_cache` | writes physical rows for `[start,start+n_tokens)` | not restored | additional pre-Pass-A snapshot required; PROVEN BY SOURCE |
| attention/index compressed cache append rows | may write future append-only rows | contents not restored; counters restored | future rows are invisible after counter restore and an emitting sequential token overwrites its current append row before use; PROVEN BY SOURCE |
| decode `cur_hc/after_ffn_hc` | generic verifier does not use them | no content restore needed | PROVEN BY SOURCE |
| batch HC/QKV/FFN scratch | overwritten; batch HC pointers ping-ponged | not restored | ordinary sequential reference uses decode scratch, not batch scratch; PROVEN BY SOURCE |
| `dspark_target_hidden(_batch)` | overwritten in capture mode | not restored | capture/proposer scratch, not target forward state; PROVEN BY SOURCE |
| DSpark support-model raw cache | target verifier does not write it | no restore needed for this pass | PROVEN BY SOURCE |
| DSpark cache window metadata | no target-verifier write found | `spec_frontier_restore()` resets it | PROVEN BY SOURCE |
| `mtp_n_raw` / MTP cache | no target-verifier write found | counter restored; cache not restored | no Pass-A effect in this call graph; PROVEN BY SOURCE |
| checkpoint length | caller temporarily appends drafts | explicitly reset to `start` | PROVEN BY SOURCE |
| checkpoint storage beyond `len` | draft values remain allocated | not cleared | invisible through `len`; the reference receives forced token arguments; PROVEN BY SOURCE |
| `s->logits` | verifier writes `spec_logits`, not session logits | no restoration needed for comparison | PROVEN BY SOURCE |
| stats/scheduler metadata | session wrapper may change it | outside model S0 | diagnostic harness should isolate it or terminate after comparison; INFERRED |

### Required execution order

```text
1. snapshot target raw-KV physical rows at S0
2. spec_frontier_snapshot
3. Pass A: generic verifier
4. spec_frontier_restore
5. restore target raw-KV physical rows
6. Pass B: ordinary sequential decode with the forced proposed sequence
```

For each target layer, raw-KV snapshotting must preserve:

```text
row_i = (start + i) % raw_cap
bytes_per_row = DS4_N_HEAD_DIM * sizeof(float)
i = 0 .. n_tokens - 1
```

Ring wrap must be handled by split copies or per-row copies. The present GO is scoped to the single-device M4 Metal configuration. A later TP/CUDA extension must also audit duplicated or cross-device raw caches.

## D. Checkpoint map

| CP | Generic verifier source | Ordinary sequential source | Layout and lifetime | Conclusion |
|---|---|---|---|---|
| CP1 normalized attention input | `metal_graph_batch_attn_norm(g)` | `metal_graph_attn_norm(g)` | F32; batch `[N, DS4_N_EMBD]`, sequential `[DS4_N_EMBD]`; overwritten by the next layer | PROVEN BY SOURCE |
| CP2 Q/KV projection | `metal_graph_batch_qr(g)` plus `metal_graph_batch_kv_raw(g)` | `metal_graph_qr(g)` plus `metal_graph_kv_raw(g)` | composite F32 checkpoint: `[N,q_rank]` and `[N,DS4_N_HEAD_DIM]` versus one-row tensors | PROVEN BY SOURCE |
| CP3 compressor projection and update | `metal_graph_batch_comp_kv/sc(g)` plus per-row frontier/counters | `metal_graph_comp_kv_cur/sc_cur(g)` plus frontier/counters | F32; `comp_width` is ratio-dependent; ratio-4 indexer projection later reuses the batch projection buffers | PROVEN BY SOURCE |
| CP4 post-attention HC | `metal_graph_batch_after_attn_hc(g)` | `metal_graph_after_attn_hc(g)` | F32; `[N, DS4_N_HC*DS4_N_EMBD]` versus one row; valid before FFN | PROVEN BY SOURCE |
| CP5 complete layer output | `metal_graph_batch_next_hc(g)` before pointer swap | `metal_graph_after_ffn_hc(g)` before pointer swap | F32; `[N, DS4_N_HC*DS4_N_EMBD]` versus one row | PROVEN BY SOURCE |

### Checkpoint corrections

CP3 must contain both the projection output and the post-update frontier/counter. The canonical Metal path may use fused projection-and-store, so it is not valid to claim that the projection checkpoint always precedes every persistent mutation.

CP4 is defined as **post-attention HC**, not raw `attn_out`. Default canonical decode may fuse attention output projection with HC expansion and therefore may not independently materialize a comparable `attn_out`. Disabling that fusion would replace the production reference path with a different execution path.

## E. Exact-N2 sanity viability

`metal_graph_verify_decode2_exact()` is source-comparable as an auxiliary rows-0/1 oracle because it:

- creates distinct HC row views for token 0 and token 1;
- calls `metal_graph_encode_decode_layer()` once per token per layer;
- uses the same positions and single-token raw spans as ordinary decode;
- saves and restores decode HC pointer identity and active tier;
- does not replace ordinary sequential decode as the full-block reference.

This semantic comparability is **PROVEN BY SOURCE**. Runtime bit identity remains **UNKNOWN**.

Before each oracle pass, frontier and raw KV must be restored to the same S0 again. If ordinary sequential and exact N=2 disagree:

1. stop using exact N=2 as an oracle;
2. investigate storage address, layout, pointer binding, and kernel selection;
3. retain ordinary sequential decode as the authority for generic-vs-sequential comparison.

## F. GPU copy primitive analysis

On Metal, `ds4_gpu_tensor_copy()`:

- requires an active command buffer;
- validates source and destination bounds;
- closes the current compute encoder and records an `MTLBlitCommandEncoder` copy;
- preserves command order within the command buffer;
- performs no CPU tensor readback;
- becomes complete before CPU access when `ds4_gpu_end_commands()` commits and waits.

It is suitable for raw-KV and checkpoint snapshotting on the single-device M4 experiment: **PROVEN BY SOURCE**.

Snapshot destinations must be dedicated non-aliasing GPU buffers. `ds4_gpu_tensor_read()` may be used only after the pass-level command buffer has completed.

## G. Minimal implementation plan and gate

Future changes remain phase-separated:

1. Add a GPU-independent bit comparator and deterministic CPU unit test. It must not include `ds4_gpu.h`.
2. Add diagnostic-only GPU snapshot storage and Pass-A capture without comparison or behavior changes.
3. Add S0 validation with raw-KV snapshot before Pass A and restoration before every reference/oracle pass.
4. Add ordinary sequential reference capture using the exact forced proposed sequence.
5. Add optional exact-N2 rows-0/1 sanity capture.
6. Compare after GPU completion and report the first `(row, layer, checkpoint, element)` mismatch.

Hard constraints:

- no replay removal;
- no direct generic-frontier commit;
- no acceptance or scheduling changes;
- no CPU readback inside the layer loop;
- no assumption that CP3 is the first mismatch;
- probe-disabled execution must retain the existing call graph and behavior.

## Final GO / NO-GO

**GO** for implementation of the corrected diagnostic differential probe.

**NO-GO** for replay removal or verifier-frontier commitment.

C0 produced no `PROVEN BY TEST` conclusion. Runtime bit identity, first-divergence position, and probe non-perturbation remain to be established in later phases.
