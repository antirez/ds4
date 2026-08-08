# Codex C2b Implementation Handoff

Read first:

1. `skill/FIRST-DIVERGENCE-EXPERIMENT-SPEC-V1.md`
2. `skill/CODEX-P2-C0-AUDIT.md`

Base branch: `codex/p2-first-divergence`

Required baseline ancestor: `34f858fce638dff779c883ada3354339cd07a95d`.

The experiment design is frozen. Do not redesign checkpoints or infer root cause.

## Task

Implement **C2b only**: Pass-A diagnostic capture plus proof that capture is non-perturbing.

Do not implement Pass B, C4, C5, replay removal, compressor exactification, acceptance changes, or optimization.

## Before coding

For each proposed checkpoint source in the frozen spec, verify at the current source baseline:

- exact producer;
- exact dtype;
- logical shape and row stride;
- generic row-to-token mapping;
- first point at which the value is valid;
- first later overwrite/reuse;
- whether the sequential candidate has exactly the same semantic meaning.

For C2b, only generic hooks are implemented, but this verification prevents capturing the wrong object now and discovering it after C4.

If any frozen candidate is semantically invalid, do not substitute another object. Record the conflict and stop that checkpoint for design review.

## Required C2b functionality

Implement a diagnostic-only path that performs:

```text
S0 snapshot
A0 generic verifier, capture OFF
record post-A observables

restore exact S0
A1 generic verifier, capture OFF
record post-A observables

restore exact S0
A2 generic verifier, capture ON
record post-A observables + checkpoint snapshots

CPU bitwise compare after GPU completion
```

### Exact S0 repetition

Use `spec_frontier_snapshot()` / `spec_frontier_restore()` plus explicit target raw-KV physical-row snapshot/restore.

Before A0, preserve original target raw-KV rows for every target layer and every proposal logical position:

```text
physical_row = (start + i) % raw_cap
row_bytes = DS4_N_HEAD_DIM * sizeof(float)
```

The same original bytes must be restored before A1 and again before A2. Do not snapshot the already-mutated A0/A1 rows as the next baseline.

Correctly handle ring wrap and reject ambiguous intra-span physical aliasing if exact S0 restoration cannot be represented safely.

### Capture transport

Use the already tested `ds4_gpu_tensor_copy_f32_inline()` for F32 checkpoint copies.

No blit fallback.
No CPU readback in the layer loop.
No command-buffer completion in the layer loop solely for capture.
No canonical kernel/fusion changes.

### Required post-A observables for A0/A1/A2 equality

At minimum compare bitwise:

- verifier/spec logits used by block verification;
- target raw-KV rows written by the verifier;
- attention compressor `state_kv`, `state_score`, `layer_n_comp`;
- index compressor `state_kv`, `state_score`, `layer_n_index_comp` where applicable;
- compressed-cache rows newly emitted and future-visible under the resulting counters;
- any additional target state discovered during implementation that survives Pass A and can affect future target decode.

Exclude only dedicated diagnostic snapshot buffers.

### Gate result

The diagnostic must explicitly print separate results:

```text
C2B_CONTROL A0_vs_A1 PASS|FAIL
C2B_PROBE   A0_vs_A2 PASS|FAIL
C2B_RESULT  PASS|FAIL
```

`C2B_RESULT PASS` requires both comparisons bit-exact across all required observables.

On failure, print object name, layer/row when applicable, mismatch count, first mismatch index/raw bits, max abs/rel/ULP using the existing comparator where applicable.

## Expected implementation discipline

- Diagnostic mode disabled by default.
- Probe-disabled production call graph/behavior unchanged.
- Prefer small diagnostic structs/helpers over invasive changes to model execution.
- Snapshot destinations must be dedicated, non-aliasing buffers.
- Do not clear or rewrite source tensors to simplify comparison.
- Do not add an alternate verifier.

## Deliverables

One reviewable commit containing:

1. C2b implementation;
2. any focused model-free/unit tests possible without the GGUF;
3. a short `skill/CODEX-C2B-IMPLEMENTATION-NOTE.md` containing:
   - exact commit baseline;
   - checkpoint source verification table;
   - files/functions changed;
   - diagnostic command for M4 Max;
   - anything intentionally marked N/A or blocked.

Do not claim C2b PASS locally unless the exact M4 Max three-run protocol has actually executed. The runtime gate belongs to Qwen execution.
