# DSpark First-Divergence Experiment Spec v1

## Authority and roles

This document freezes the diagnostic experiment design. It is authoritative for C2b/C4/C5 implementation unless explicitly revised by the experiment designer.

- Experiment design / semantic adjudication: GPT-5.6 Thinking.
- Implementation: Codex only.
- M4 Max execution and raw-log collection: Qwen only.
- Qwen must not change checkpoint definitions, implementation, environment semantics, or interpret source-level differences as runtime divergence.
- Codex must not change canonical decode behavior to make a checkpoint easier to expose.

Implementation baseline: `34f858fce638dff779c883ada3354339cd07a95d`.

This baseline contains the verified CPU F32 comparator and the M4 Max-proven same-compute-encoder F32 snapshot transport.

## Non-goals / hard prohibitions

Do not:

- remove replay;
- commit generic verifier frontier directly;
- exactify compressor or other kernels;
- disable sequential fusion to manufacture symmetric temporaries;
- change scheduler / acceptance policy;
- change proposal generation;
- use exact-N2 as the canonical reference;
- read GPU tensors back to CPU inside the layer loop;
- infer numerical divergence from different source helpers;
- proceed past a failed gate.

Ordinary sequential `metal_graph_eval_token_raw_swa(...)` remains the only canonical reference.

---

# 1. Evidence policy

Use three evidence levels only:

- `PROVEN BY SOURCE`: call graph, tensor producer, layout, lifetime, restoration logic.
- `PROVEN BY TEST`: runtime result from the frozen experiment.
- `UNKNOWN`: anything numerical not directly measured.

In particular:

`different implementation != runtime numerical divergence`.

The Qwen CP1/CP2 summary is accepted only as source-call-graph evidence. Its claims that CP1 is already the fundamental or earliest numerical divergence are rejected until runtime bitwise comparison proves them.

---

# 2. S0 restoration contract — C3 freeze

Pass B is valid only if it starts from the same model-visible S0 as Pass A.

Required order:

```text
S0
  1. snapshot target raw-KV physical rows that Pass A may overwrite
  2. spec_frontier_snapshot()

Pass A: generic verifier

Restore
  3. spec_frontier_restore()
  4. restore target raw-KV physical rows from the pre-Pass-A snapshot
  5. reset checkpoint.len to start

Pass B: ordinary sequential decode of the exact forced proposal sequence
```

## MUST RESTORE

### Compressor frontiers

Use the existing speculative frontier snapshot/restore for:

- `layer_attn_state_kv`
- `layer_attn_state_score`
- `layer_n_comp`
- `layer_index_state_kv`
- `layer_index_state_score`
- `layer_n_index_comp`
- associated restored cache-window/frontier metadata already covered by `spec_frontier_restore()`

### Target raw KV physical rows

`spec_frontier_restore()` is insufficient because Pass A writes target raw-KV physical rows.

For every target layer and proposal row `i`:

```text
logical_pos  = start + i
physical_row = logical_pos % raw_cap
row_bytes    = DS4_N_HEAD_DIM * sizeof(float)
```

Snapshot these physical rows before Pass A and restore them after `spec_frontier_restore()` and before Pass B.

The implementation must correctly handle ring wrap. Per-row copies are acceptable and preferred for the diagnostic path if they make the physical mapping unambiguous.

The snapshot must preserve the original S0 bytes even if two future logical positions alias a physical row under an unusually small raw ring. If aliasing occurs within the proposal span, the implementation must either preserve enough original-row identity to restore S0 exactly or reject the diagnostic run with a clear reason. It must not silently assume the proposal span is non-aliasing.

### Logical session position

`checkpoint.len` must be reset to `start` before Pass B. Draft entries remaining in allocated storage beyond `len` are invisible and need not be zeroed.

## INVISIBLE AFTER RESTORE

Compressed-cache rows written beyond the restored `layer_n_comp` / `layer_n_index_comp` counters do not need byte restoration only if the existing source invariant remains true: after counter rollback, the stale rows are not addressable as live rows and the next append overwrites the current append row before it can be consumed.

Codex must not broaden this claim beyond the exact current call graph. If implementation inspection finds any read-before-overwrite path, promote those rows to MUST RESTORE and stop for design review.

## SCRATCH / NOT PART OF S0

No restoration is required for generic batch HC/QKV/FFN scratch when ordinary sequential Pass B uses distinct decode scratch and the batch scratch cannot be read as persistent target state.

`dspark_target_hidden*` is diagnostic/proposer capture scratch and is not canonical target forward state.

## NO PASS-A EFFECT EXPECTED

Support-model / MTP cache state is not part of the target verifier mutation set for this single-device M4 experiment unless implementation inspection discovers a write in the Pass-A call graph.

## OUTSIDE MODEL S0

Stats / scheduler bookkeeping must be isolated from the experiment result. Prefer a diagnostic entry point that terminates after comparison and does not allow these counters to influence another generation step. Do not treat scheduler/stat values as model-state checkpoints.

C3 verdict under this contract: `GO FOR DIAGNOSTIC IMPLEMENTATION`, contingent on exact raw-KV physical-row restoration being implemented and later validated at runtime.

---

# 3. Checkpoint Freeze v1

The comparison unit is `(proposal row, layer, checkpoint subobject)`.

For proposal row `r`, generic row `r` must be compared with the ordinary sequential state produced while forcing proposal token `r` at logical position `start + r`, after the same preceding forced proposal tokens `0..r-1` have been sequentially decoded.

All checkpoint payloads are F32 unless the actual source object is not F32; no conversion may be introduced merely for comparison.

## CP1 — normalized attention input

Semantic object: the normalized hidden input consumed by the attention projection for this layer/token.

Generic source: `metal_graph_batch_attn_norm(g)` row `r`.

Sequential source: `metal_graph_attn_norm(g)` for the current token.

Capture: immediately after the normalized-attention producer, before reuse/overwrite.

Purpose: prove whether both paths enter attention projection with the same logical hidden input.

## CP2-Q — Q-side projection semantic output

Semantic object: the Q projection object that is common between the two paths before downstream attention transforms that would make the semantics differ.

Current source-map candidate from C0:

- generic: row `r` of `metal_graph_batch_qr(g)`;
- sequential: `metal_graph_qr(g)`.

Codex must verify exact dtype, width, stride, producer, and first-valid point at baseline `34f858f` before adding the hook. If those objects are not semantically identical, STOP and report instead of inventing a replacement.

## CP2-KV-P — raw KV projection output before cache persistence

Semantic object: the per-token raw KV projection before it is interpreted as a persistent raw-cache row.

Current source-map candidate from C0:

- generic: row `r` of `metal_graph_batch_kv_raw(g)`;
- sequential: `metal_graph_kv_raw(g)`.

Same rule: semantic equivalence must be source-proven before hooking.

## CP2-KV-R — persistent raw-KV physical row after store

Semantic object: bytes of the raw-KV cache row corresponding to logical position `start + r` after that row has been written by the path under test.

Physical mapping:

```text
physical_row = (start + r) % raw_cap
```

This checkpoint is distinct from CP2-KV-P. A mismatch here with CP2-KV-P exact localizes the first difference to persistence / mapping / store semantics rather than projection arithmetic.

Capture timing:

- generic: after the batch raw-KV store has completed encoding for the relevant rows;
- sequential: after the ordinary one-row raw-KV store for the current token;
- before a later operation can overwrite the row.

## CP3-P — compressor projection semantics — CONDITIONAL

Generic has naturally materialized `batch_comp_kv` / `batch_comp_sc`.

Canonical sequential Metal may use fused projection-and-state-store. CP3-P is enabled only if Codex can source-prove a physical sequential object whose bytes represent exactly the same mathematical projection object as the generic tensor without disabling fusion, changing kernel selection, inserting recomputation, or moving a persistent-state mutation.

If this proof is not available, CP3-P is OMITTED from v1. This is a valid outcome.

No experiment may change the sequential path merely to obtain CP3-P symmetry.

## CP3-F — persistent compressor frontier — MANDATORY where compressor exists

Semantic object: future-visible compressor state after the normal compressor update point for the current path.

Capture at least:

- attention `state_kv`;
- attention `state_score`;
- `layer_n_comp`;
- index `state_kv` / `state_score` / `layer_n_index_comp` on layers where the index compressor participates;
- newly emitted compressed-cache row when an emit occurs and that row is part of future-visible state.

Comparison must use the same semantic frontier after processing proposal prefix `0..r`.

Layers with no compressor state must record `N/A`, not fabricate zero tensors.

## CP4 — post-attention HC

Semantic object: post-attention hidden/carrier state immediately before FFN processing.

Current source-map candidate from C0:

- generic: row `r` of `metal_graph_batch_after_attn_hc(g)`;
- sequential: `metal_graph_after_attn_hc(g)`.

Do not replace this with a convenient raw `attn_out` if canonical sequential fusion does not materialize an equivalent object.

## CP5 — complete layer output

Semantic object: final layer HC output immediately before the path's pointer swap / next-layer handoff.

Current source-map candidate from C0:

- generic: row `r` of `metal_graph_batch_next_hc(g)` before pointer swap;
- sequential: `metal_graph_after_ffn_hc(g)` before pointer swap.

---

# 4. C2b — probe non-perturbation gate

C2b implements PASS A ONLY. Do not implement Pass B in the same commit.

Goal: prove that checkpoint capture does not alter generic verifier results or future-visible state.

## Required three-run protocol

All runs must start from equivalent S0 and use the exact same proposal sequence.

```text
A0: capture OFF
restore exact S0
A1: capture OFF
restore exact S0
A2: capture ON
```

### Gate 1 — deterministic control

Compare A0 vs A1 bitwise for the Pass-A observables below.

If A0 != A1: STOP. The experiment is nondeterministic under the current harness and probe perturbation cannot be interpreted.

### Gate 2 — non-perturbation

Compare A0 vs A2 bitwise.

Required Pass-A observables:

- verifier/spec logits used to judge the proposal block;
- raw-KV rows written by Pass A;
- attention compressor state tensors and counters;
- index compressor state tensors and counters where applicable;
- compressed-cache rows newly emitted by Pass A and future-visible under the resulting counters;
- any other target state Codex discovers that survives the verifier call and can influence future ordinary target decode.

The dedicated checkpoint snapshot buffers themselves are excluded.

C2b PASS requires every required observable to be bit-identical.

If capture ON changes any observable: STOP. Do not implement C4/C5 until the perturbation is resolved.

The snapshot copy must use the already validated `ds4_gpu_tensor_copy_f32_inline()` for eligible F32 objects. No blit fallback may be silently substituted for checkpoint capture.

---

# 5. C4 — canonical forced-token Pass B

Implement only after C2b runtime PASS.

Pass B must:

1. restore the exact S0 contract above;
2. take the proposal token IDs generated for Pass A as immutable input;
3. call ordinary canonical sequential target decode for token 0, token 1, ... in that exact order;
4. never regenerate proposal tokens with argmax;
5. capture frozen checkpoints for each token/layer at the canonical producer boundaries;
6. preserve ordinary sequential kernel/fusion selection.

The proposal sequence is an experimental input, not an output of Pass B.

Exact-N2 may be run as a separate auxiliary sanity oracle only after another exact S0 restore. Disagreement with exact-N2 invalidates that auxiliary oracle, not the ordinary sequential reference.

---

# 6. C5 — first-divergence comparison

After both passes complete and GPU work is synchronized, compare snapshots on CPU with the verified F32 comparator.

The report must be ordered by execution semantics:

```text
row ascending
  layer ascending
    CP1
    CP2-Q
    CP2-KV-P
    CP2-KV-R
    CP3-P if enabled
    CP3-F subobjects
    CP4
    CP5
```

For each compared object report:

- exact / mismatch;
- element count;
- mismatch count;
- first mismatching element index;
- actual/expected raw bits for the first mismatch;
- max absolute difference when defined;
- max relative difference when defined;
- max ULP distance when defined.

The first mismatch in this semantic order is the `first-divergence candidate`.

Do not label a root cause solely from the checkpoint name. Example:

- `CP2-Q first mismatch` localizes divergence to the interval after CP1 and at/before the Q projection output; it does not by itself prove a specific Metal kernel instruction is the root cause.
- `CP2-KV-P exact + CP2-KV-R mismatch` strongly localizes to persistence/mapping/store semantics.
- `CP2 exact + CP3-F mismatch` supports the compressor-update/frontier hypothesis even if CP3-P is unavailable.

A later narrow operator test is required before production exactification.

---

# 7. Codex implementation sequence

Codex must produce separate reviewable commits.

## Commit C2b-1 — semantic hook verification + Pass-A capture

- verify each frozen source object against `34f858f`;
- record any correction in an implementation note;
- implement capture buffers and hooks for Pass A only;
- implement raw-KV S0 snapshot/restore utility needed to repeat A0/A1/A2 from exact S0;
- implement A0/A1/A2 non-perturbation diagnostic entry point;
- no Pass B;
- no runtime first-divergence claim.

If a frozen checkpoint candidate is semantically invalid, Codex must stop that checkpoint and report the conflict. It must not redesign the checkpoint.

## Qwen execution gate C2b

Qwen builds and runs C2b on the M4 Max. It returns:

- exact commit SHA;
- `git status --short`;
- build command and full exit status;
- machine / Metal device identification;
- exact environment variables;
- complete A0/A1/A2 diagnostic log;
- PASS/FAIL as printed by the program, without reinterpretation.

No source edits are permitted during execution.

## Commit C4/C5-1 — sequential reference + comparison

Only after C2b PASS:

- implement exact S0 restore before Pass B;
- implement forced-token ordinary sequential Pass B;
- capture frozen sequential checkpoints;
- implement post-run CPU comparison and first-divergence report;
- keep replay and production behavior untouched outside diagnostic mode.

## Qwen execution gate C5

Qwen runs the frozen command on M4 Max and returns the raw full log plus a machine-readable summary if the program emits one. Qwen does not diagnose or alter the experiment.

---

# 8. Stop conditions

Immediately stop and return to design review if any of the following occurs:

- A0 != A1;
- A0 != A2;
- exact raw-KV S0 restoration cannot be demonstrated for the proposal span;
- a checkpoint requires changing canonical sequential kernel selection;
- proposal tokens differ between Pass A input and Pass B forced input;
- Pass B does not call the ordinary sequential target path;
- a snapshot source is overwritten before the inline copy is encoded;
- GPU readback/synchronization is inserted inside the layer loop;
- any implementation path silently falls back to a different copy or compute primitive.

---

# 9. Current gate state after this freeze

```text
C0       CLOSED
C1       PASS — PROVEN BY TEST
C2-pre   CLOSED / CONDITIONAL GO
C2a      PASS — PROVEN BY TEST on M4 Max
CP1→CP2  SOURCE SEMANTICS ACCEPTED WITH CORRECTION:
         implementation differences proven; numerical divergence unknown
CP3      SOURCE CALL GRAPH CLOSED
C3       GO FOR DIAGNOSTIC IMPLEMENTATION under exact raw-KV restore contract

Checkpoint Freeze v1  FROZEN by this document

C2b      NEXT — Codex implementation, then Qwen M4 Max execution
C4       BLOCKED until C2b runtime PASS
C5       BLOCKED until C2b runtime PASS and C4 implementation
```
