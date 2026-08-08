# AGENTS.md — DSpark exactness investigation contract

This file is the short operational contract for coding/review agents working on the DSpark exactness investigation.

It intentionally freezes the current plan so agents do not drift as context grows.

For deeper source analysis, also read:

- `skill/CODEX-P2-C0-AUDIT.md`
- any later phase-specific audit documents on this branch

Do not use `backup/qwen-p2-draft` as an implementation base. It is historical evidence only.

---

## 1. Research goal

The current goal is **not** to remove replay.

The current goal is to locate the earliest tensor/state divergence between:

1. the generic DSpark batched verifier; and
2. canonical ordinary sequential target decode,

when both begin from the same effective state S0 and consume the same forced proposed token sequence.

Ordinary sequential target decode through `metal_graph_eval_token_raw_swa(...)` is authoritative.

`metal_graph_verify_decode2_exact()` is auxiliary only.

Do not assume the compressor is the first divergence.

---

## 2. Priority order

When goals conflict, use this order:

1. correctness;
2. experimental validity;
3. reproducibility;
4. root-cause localization;
5. maintainability;
6. performance.

A faster path that cannot prove canonical accepted-prefix state is not acceptable.

---

## 3. Upstream-first rule

DS4 changes quickly.

Before every phase that touches GPU execution, verifier behavior, state restoration, or production code, check current `antirez/ds4` upstream first.

At minimum inspect:

- current `main`;
- recent DSpark PRs;
- relevant DSpark issues;
- closed/unmerged experimental PRs.

Historical references that must remain in mind:

- issues / PRs around #658 and #659: greedy-identity break caused by noncanonical verifier state;
- PR #590: replay-free prefix checkpoints and same-compute-encoder F32 copy mechanism; copy mechanism is useful historical evidence, replay-free correctness claims are not authoritative for this project;
- PR #677: byte-exact Metal verifier, exact but too slow as a general solution;
- PR #670: ROCm verifier optimization ideas, not direct Metal exactness evidence.

If upstream changes any relevant call graph or already implements the intended primitive, stop and report the delta before duplicating work.

Current source baseline used by C0/C1/C2-pre analysis:

`b0309611041655f4e45671cfd9c9886aff161406`

Do not assume this SHA is still current without checking.

---

## 4. Evidence labels

Every important conclusion should be labeled mentally or explicitly as one of:

- `PROVEN BY SOURCE`
- `PROVEN BY TEST`
- `INFERRED`
- `HYPOTHESIS`
- `UNKNOWN`
- `REJECTED`

Never promote an inference or hypothesis into a fact because a function name contains words such as `exact`.

---

## 5. Frozen phase status

### C0 — static experiment-design audit

Status: **CLOSED / GO for diagnostic investigation only**.

C0 does **not** authorize replay removal or verifier-frontier commit.

Important frozen conclusions include:

- ordinary sequential decode is authoritative;
- raw target KV physical rows are not fully restored by `spec_frontier_restore()` and must be handled explicitly in S0 restoration;
- CP1–CP5 must be semantic boundaries, not merely convenient tensor names;
- CP4 is post-attention HC, not forced raw `attn_out`;
- exact-N2 is auxiliary only;
- probe non-perturbation requires runtime validation.

### C1 — pure CPU exact float comparator

Status: **CLOSED / PROVEN BY TEST**.

Commit:

`2596a70ade50e87206a4995593ff7e1887d2459e`

`make test-float-compare` passed.

C1 is GPU-independent and must stay GPU-independent.

### C2-pre — same-encoder capture feasibility audit

Status: **CLOSED / CONDITIONAL GO**.

Findings:

- current Metal baseline already contains the F32 copy shader, pipeline, and internal same-compute-encoder helper needed for inline F32 snapshot transport;
- PR #590 mainly added a public wrapper around machinery already present in current main;
- CP1–CP5 approved GPU checkpoint payloads are contiguous F32 and do not inherently require a blit;
- host counters such as `layer_n_comp` are CPU scalars and must be saved separately;
- silent fallback from inline compute copy to blit is unacceptable for the diagnostic path;
- same-encoder inline copies still require runtime non-perturbation validation.

---

## 6. New phase split: C2a and C2b

The old single C2 phase is superseded.

### C2a — validate the checkpoint transport primitive

**CURRENT IMPLEMENTATION PHASE.**

Goal: prove that same-compute-encoder F32 snapshot copy is trustworthy before it is inserted into the DSpark verifier.

C2a may implement only the minimum inline F32 copy wrapper and focused regression tests.

Required C2a properties:

1. no new Metal shader if existing `kernel_cpy_f32_f32` can be reused;
2. no silent blit fallback in the tested diagnostic path;
3. dedicated non-overlapping source/destination buffers;
4. bit-preservation tests using explicit raw `uint32_t` patterns;
5. authoritative comparison with the C1 comparator;
6. source buffer remains bit-identical after copy;
7. ordinary Metal build remains link-safe;
8. no DSpark/verifier/replay/checkpoint hooks yet.

Required bit-pattern coverage includes at least:

- ordinary positive/negative finite values;
- `+0` / `-0`;
- `+Inf` / `-Inf`;
- distinct NaN payloads;
- positive/negative subnormals;
- values around the normal/subnormal boundary.

Important: current `kernel_cpy_f32_f32` is typed `float -> float`, not explicit `uint32_t -> uint32_t` transport. Therefore NaN payload, signed-zero, and subnormal preservation must be proven at runtime rather than assumed from source.

If any tested raw bit pattern changes, **STOP**. Do not redesign/fix the Metal kernel in the same C2a commit.

C2a exit decision must be exactly one of:

- `C2a PASS — ready for checkpoint freeze / C2b`
- `C2a FAIL — copy primitive not trustworthy`
- `C2a BLOCKED — infrastructure assumption invalid`

After C2a, **STOP. Do not start C2b automatically.**

### C2b — Pass-A checkpoint instrumentation + non-perturbation proof

Status: **BLOCKED until checkpoint freeze**.

Do not implement C2b until the CP1→CP3 semantic audits are reviewed and the exact capture locations are frozen.

C2b will later:

- allocate dedicated snapshot buffers;
- capture approved CP1–CP5 semantic checkpoints on Pass A;
- use same-encoder compute copies where appropriate;
- save CPU counters separately;
- avoid synchronous CPU tensor reads inside the layer loop.

C2b exit gate requires a controlled `capture OFF` vs `capture ON` experiment from equivalent S0.

The probe-on path must be bit-identical to probe-off for all relevant verifier outputs and live state that future execution can observe.

If probe-on changes the computation being measured, **STOP** and redesign the probe.

Even same-encoder copies are not assumed non-perturbative merely because they avoid blit encoder transitions.

---

## 7. Checkpoint semantics currently under independent audit

Do not hard-code capture locations until these are frozen.

Current coarse semantic checkpoints are:

- **CP1** — normalized attention input;
- **CP2** — Q projection + raw KV projection/state at the comparable post-projection boundary;
- **CP3** — compressor projection/update persistent state boundary; exact observable decomposition is still being audited;
- **CP4** — post-attention HC;
- **CP5** — complete layer output HC before pointer swap/reuse.

Independent Qwen audits are currently mapping:

- layer input → CP1 → CP2;
- CP2 → CP3 compressor/indexer state machine.

Those audits are source maps only. They must not make runtime exactness claims.

C2b must wait for their review and a checkpoint freeze.

---

## 8. Later phases

### C3 — S0 restoration validation

Goal: prove Pass B can start from the same effective target-model state as Pass A.

Required elements include:

- pre-Pass-A raw-KV physical-row snapshot;
- `spec_frontier_restore()`;
- raw-KV physical restore;
- ring-wrap handling;
- causal proof that any non-restored scratch/append-only content is invisible before overwrite/reinitialization.

If S0 cannot be restored exactly enough, **STOP**.

### C4 — canonical ordinary sequential reference

Run the exact same proposed token sequence through ordinary single-token target decode from restored S0.

Capture the same semantic checkpoints as Pass A without altering the canonical reference path.

Exact-N2 rows 0–1 may be used only as an auxiliary oracle.

### C5 — first-divergence localization

Compare generic batch vs ordinary sequential after GPU completion.

Required report includes:

- row;
- layer;
- checkpoint;
- first mismatching element;
- mismatch count;
- max absolute difference;
- max relative difference;
- max ULP.

The successful C5 outcome is the earliest reproducible mismatch interval.

Then subdivide only that interval.

Do not broadly instrument the entire graph again.

---

## 9. Hard stop / forbidden work

Until C5 evidence supports otherwise, do **not**:

- remove replay;
- directly commit generic verifier frontier state;
- add or reintroduce `DS4_DSPARK_EXACT_NOREPLAY`-style behavior;
- weaken greedy-identity criteria;
- change acceptance policy, scheduler, confidence, or block size to hide verifier cost;
- design an exact compressor kernel before the first-divergence experiment localizes the problem there;
- use CPU recomputation as the canonical Metal reference;
- force-disable fusion merely to expose a convenient tensor;
- treat `worst_argmax_gap=0` as proof of hidden-state bit identity;
- cherry-pick PR #590 wholesale;
- continue automatically into the next phase without an explicit gate review.

---

## 10. Git discipline

Use:

`one hypothesis / one focused change / one benchmark or test / one commit`

Do not mix diagnostic infrastructure, production optimization, replay changes, and unrelated refactors in one commit.

Never discard or overwrite existing user work.

Before committing, inspect at minimum:

- `git status --short`
- `git diff --stat`
- the full relevant diff

After a phase-specific commit, stop and report results rather than starting the next phase.

---

## 11. Final success criteria

The project is not successful merely because a local kernel is faster or final text often matches.

A replay-free optimization is only eligible after all of the following are demonstrated:

1. the corrected state-critical operator/state boundary is bit-identical to ordinary sequential decode;
2. accepted-prefix committed state equals ordinary sequential state for all future-visible model state;
3. long `--temp 0` generations are byte-identical across multiple prompts;
4. acceptance correctness is unchanged;
5. replay is genuinely removed or materially reduced, not shifted elsewhere;
6. end-to-end generation throughput beats the plain target baseline on the intended workload;
7. the non-DSpark base path does not regress.

Until then, the project remains a diagnostic exactness investigation.
