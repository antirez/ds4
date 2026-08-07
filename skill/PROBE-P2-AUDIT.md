# P2 Audit Results — Dual-Execution Differential Probe

## P2 Prerequisite Audits (completed 2026-08-08)

### Critical architectural findings from tracing spec_frontier / verifier path

---

### Audit 1: State Restoration — `spec_frontier_snapshot/restore` completeness

**Question**: Does the existing snapshot/restore machinery return every mutable state to pre-speculation S0?

| Mutable state | Modified by generic verifier? | Captured by snapshot? | Restored by restore? | Needs extra probe snapshot? |
|--------------|-----------------------------|----------------------|---------------------|----------------------------|
| compressor KV frontier (per layer) | ✅ | `spec_attn_state_kv[il]` full tensor copy | `→ layer_attn_state_kv[il]` full tensor copy | ❌ no |
| compressor score frontier (per layer) | ✅ | `spec_attn_state_score[il]` full tensor copy | `→ layer_attn_state_score[il]` full tensor copy | ❌ no |
| indexer KV frontier (ratio=4 layers) | ✅ | `spec_index_state_kv[il]` full tensor copy | `→ layer_index_state_kv[il]` full tensor copy | ❌ no |
| indexer score frontier (ratio=4 layers) | ✅ | `spec_index_state_score[il]` full tensor copy | `→ layer_index_state_score[il]` full tensor copy | ❌ no |
| compressed row counters (`layer_n_comp`, `layer_n_index_comp`) | ✅ | saved in `f->n_comp[]`/`f->n_index_comp[]` | restored to struct fields | ❌ no |
| DSpark cache window (`cache_start`, `cache_token_start`, `cache_len`) | ✅ | saved in `f->dspark_cache_*` | restored via `metal_graph_dspark_cache_set_window()` | ❌ no |
| mtp_n_raw counter | ✅ | saved in `f->mtp_n_raw` | restored to struct field | ❌ no |
| raw KV cache contents at spec positions [start..start+n-1] | ✅ (via `metal_graph_decode_kv_store`) | ❌ NOT captured | ❌ NOT restored | ⚠️ does not matter — replay overwrites them |
| HC state (cur_hc / after_ffn_hc) | ❌ not modified by generic verifier | N/A | N/A | ❌ no |
| checkpoint.len / token position | ❌ (CPU struct, unchanged) | N/A | N/A | ❌ no |

**Conclusion**: `spec_frontier_restore()` **fully returns the graph to pre-speculation state S0** for all GPU-visible mutable state. The only gap is raw KV cache dirty writes, but replay's sequential single-token calls overwrite those positions in order, so this gap is benign.

**P2 implication**: After verifier + restore, the graph IS at S0 (compressed frontiers clean). We can safely run canonical reference pass from same S0.

---

### Audit 2: Tensor Lifetime & Probe Integration Points

**Question**: Where in existing tensors can we inject probe snapshots without adding GPU synchronization?

#### Existing GPU snapshot storage (already allocated, no new memory needed):
```
g->spec_prefix1_attn_state_kv[il]    — per-layer compressor KV output at slot N
g->spec_prefix1_attn_state_score[il]  — per-layer compressor score output at slot N
g->dspark_target_hidden_batch         — batch HC capture (already used for layers 40,41,42)
```

#### Key insight: `metal_graph_dspark_capture_verified_suffix_layer()` (line 26525)
This function is ALREADY called per layer inside the generic verifier loop (line 34525). It:
- Copies `metal_graph_batch_cur_hc(g)` → GPU storage via `ds4_gpu_hc_weighted_sum_tensor`
- Does this as a **GPU→GPU tensor copy** (no CPU synchronization)
- Only captures last row's HC

**This is the exact injection mechanism for probe hooks**: extend this per-layer hook to also capture the 5 coarse checkpoints by adding GPU→GPU copies to dedicated probe buffers. No `ds4_gpu_tensor_read()` inside the loop.

#### Tensor visibility during verifier execution:
- `metal_graph_batch_cur_hc(g)` — visible after each layer's full batch encode
- `g->layer_attn_state_kv[il]` / `g->spec_attn_state_score[il]` — intermediate states during attention encoder
- `g->layer_raw_cache[il]` — written by `metal_graph_decode_kv_store`, overwritten between layers

#### Critical: no synchronous tensor_read inside layer loop
All probe captures must use GPU→GPU copies only. CPU readback/comparison happens after the pass completes via `ds4_gpu_end_commands()` + batched reads.

---

### P2 Architecture Decision: dual-pass, not single-pass

**Why exactifier alone is insufficient**:
- Only handles N=2 (two tokens), DSpark block_size=5 means n_tokens=6
- Exactifier and generic verifier are **separate calls** in different code paths
- They don't share intermediate state during execution
- Even if they did, exactifier's per-row HC is overwritten by the next row's computation

**Why ordinary sequential decode as canonical reference**:
- For DSpark N=5: need to compare verifier output at all 6 positions, not just 0 and 1
- `metal_graph_eval_token_raw_swa()` is the authoritative canonical path for single-token execution
- Forces same draft tokens through per-row execution = golden reference

**Execution model**:
```
=== Pass A (existing cycle) ===
1. spec_frontier_snapshot(&frontier, s)     → captures S0 (compressed frontiers)
2. push drafts to checkpoint                → [d0, d1, ..., d5]
3. metal_graph_verify_suffix_tops()         → generic batched verifier
   └─ probe: GPU→GPU copy each layer's 5 coarse checkpoints to snapshot buffers
4. spec_frontier_restore(&frontier, s)      → restores compressed frontiers

=== Pass B (canonical reference) ===
5. Capture raw KV cache position info from S0
6. Run metal_graph_eval_token_raw_swa() for d0, then d1, ..., d5 sequentially
   └─ probe: GPU→GPU copy each layer's 5 coarse checkpoints to snapshot buffers

=== Comparison (CPU-only) ===
7. ds4_gpu_end_commands() + readback
8. byte-for-byte compare batch[cp] vs ref[cp] for all layers, all checkpoints
9. Report first divergence with probe info
```

**P2 prerequisite check — S0 definition**: 
S0 = (checkpoint[start], spec_frontier_snapshot data). All GPU mutable state is returned to pre-speculation values by step 4. Pass B starts from same S0. ✓

**Probe buffer design**: Each checkpoint needs per-layer per-row storage for N tokens. Since probe only runs during investigation (not production), can allocate probe buffers as part of `ds4_gpu_graph` during P2 activation.

---

### Cleanup of premature work
The file `ds4_exactness_probe_api.h` (216 lines) was written before these audits — **it should be abandoned**. Its approach was wrong: inline CPU reads inside the layer loop, and a standalone API header instead of leveraging existing `metal_graph_dspark_capture_verified_suffix_layer()` as the hook mechanism.

---

### Coarse checkpoints (5 total)
1. After HC preparation / attention norm  — before Q/KV projection
2. After Q/KV projections                 — after q_path + kv_path matmuls
3. After compressor projection/update      — after compressor pair matmul + update
4. After attention output                 — after attn_output matmul
5. After complete layer / HC update       — after FFN, before batch_cur_hc swap

Expected first result: CP1=exact, CP2=exact, CP3=mismatch → narrows to compressor.
