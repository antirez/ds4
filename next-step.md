
## 2026-08-29 — GLM 5.3 Flash bring-up on this branch (M5 Max 128GB)

Worktree branch `glm53-flash` at upstream `a60a2a0`. Q2 GGUF (90 GiB, 96.5 GB).
First-run state, greedy `--temp 0`, ctx 8192, ~33-token prompt, `-n 256`:

| mode | prefill | decode |
|---|---|---|
| serial | 76.2 ±0.3 t/s | 33.0 t/s |
| --mtp | 76.2 t/s | 40.5 t/s |

- MTP accept rate 89.6% (135 cycles), verify2 median 37.4 ms, head+draft ~6 ms.
- Internal MTP cycle math says 40.9 t/s; wall says 40.5 — consistent.
- Greedy text serial == MTP byte-identical at short prompt; differs at a
  4500-token pangram prompt (near-tie flip, documented).
- Serial decode 33 t/s vs MTP 40.5 — only 1.23×. Verify (2-token batch
  forward) median 37.4 ms vs serial decode ~30 ms/token: verify is nearly
  2× serial cost, eating most of the accept-rate win.


### Decode stage profile (DS4_METAL_DECODE_STAGE_PROFILE=all, 31 tokens, profiling-inflated)

Shares across all 46 layers (relative, profiled wall ~109 ms/token vs ~30 ms real):
attn_output 17.9%, routed_moe 14.3%, ffn_norm 10.2%, attn_norm 8.8%,
shared_gate_up_swiglu 8.7%, residual 8.2%, router 7.9%, shared_down 7.5%.
The stage boundaries carry their own CPU cost (each ~0.2 ms stage = one
command-buffer boundary under profiling), so the profile exaggerates the
small-kernel stages. Structural facts that remain actionable:

- Layers 0-2 are dense FFN (`dense_gate_up_swiglu` 0.37-0.48 ms + `dense_down`
  0.28-0.31 ms); every other layer is router+routed_moe (8/288 experts).
- KDA layers (0,1,2,4,5,6,8,...) collapse the entire attention into one
  fused `kernel_glm53_kda_decode` (attn_norm + attn_output bracket it).
- mHC bookkeeping (hc_pre/expand/contract) runs between every stage pair
  and is folded into the 0.2-0.25 ms norm/residual buckets.

### Longer decode runs (4500-token pangram prompt, n=512, ctx 8192)

serial 29.70 t/s, --mtp 29.26 t/s — at long post-prefill context the MTP
cycle's fixed overhead (verify ~37 ms + head/draft ~6 ms for 2 tokens ≈
21.7 ms/tok best case) sits barely below serial (~33 ms/tok), and any
reject drags the average under serial. Accept rate stays ~90%.
=> Decode win is 1.0-1.2x at n=512, 1.23x at n=256. The binding constraint
is verify cost (2-token full forward ≈ 37 ms vs serial 30 ms), not acceptance.

Profiling overhead note: stage profile forces end/begin commands at each
boundary (~0.2 ms each, 46 layers x ~8 boundaries), inflating 30 ms/token
to 109 ms/token. Shares are only indicative; boundary count per stage is the
bias, so "attn_output 17.9%" ≈ "the matmul after attention", etc.

### Decode flush interval sweep (serial, n=128, short prompt)
default(4 indexed) 33.46; 4: 33.85; 8: 33.76; 16: 33.57; 32: 33.26 t/s.
No lever — decode is not flush-bound. Left default.


### Indexed-decode A/B at ctx 8192 (pos < 4096 full-attn cap)
Wait: log says "compact indexed decode is used beyond the cap", and decode
used full KV below cap already; toggling the gate moved nothing (33.3 both,
40.9 both). Not a lever at this ctx.

### The MTP verify math
Serial ~33 t/s = 30.3 ms/tok. MTP accept-cycle: verify(2-tok) 37.4 +
head/draft 5.9 = 43.3 ms per 2 tokens = 21.7 ms/tok ideal; at 89.6% accept,
mean = 0.896*21.7 + 0.104*(43.3+30.3 replay) ≈ 27.1 ms/tok ≈ 36.9 t/s.
Measured 40.5 — replay on reject is cheaper than a full serial token because
the reject path reuses saved KDA state + 1-token forward (~20 ms).
=> To go meaningfully faster: cut verify2 cost (batch-of-2 forward), cut
head+draft (6 ms), or extend verify width to 3-4 drafts.

### Verify-width ceiling
--mtp-draft is ignored on the GLM path (width fixed at 2; accepted_cap gate).
Widening means generalizing ds4_session_glm_spec_cycle_impl to K-draft trees
and glm_graph_forward_tokens(n_tokens=K) — bigger change, saved for after
cheaper wins.

### Prefill MoE path finding (2048-tok chunk, IQ2 routed)
- Prefill shares: routed_moe 30.9%, attn_output 25.4%, kv_path 14.7%.
- GLM53 Q2 uses gate IQ2_XXS + down Q2_K; the fused
  kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16 (gate+up+SwiGLU in one grouped
  MMA pass, f16 mid) is gated on n_expert==6 (DSV4's 6/256); GLM53 uses
  8/288 so it runs separate gate GEMM + up GEMM + SwiGLU + more traffic.

### Tried: relaxing n_expert==6 gate for kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16
Golden byte-identical. Prefill ABBA (4500-tok prompt): PAIR 358/290/296,
SEP 313/301/304 — median pair 296 vs sep 304. No win at 8/288 (work list
grouping amortizes the fused epilogue only at DSV4's 6-of-256 fanout).
Reverted. Exact-logits harness `test-metal-session-batch` fails identically
on unmodified a60a2a0 (max_abs 8.2e-05, tolerance 0) — pre-existing upstream
state, not caused by this change.

### MTP verify profile (2-token forward, profiling-inflated)
Verify2 = 37.4 ms real for TWO rows vs serial 30 ms for one row. Per-layer:
verify 0.80 ms vs serial 0.65 ms. The 2-row batch is nearly free in FLOPs —
decode is dispatch/latency-bound (~10 kernel launches/layer/token, small
GEMV shapes). Consequence: the fastest road to big decode t/s is wider
speculation (K drafts per verify), because verify width 2..4 costs nearly
the same wall time. Matches the Qwen finding (n_tok=2 near-flat).

### Decode is launch-bound (dispatch-bound)
Evidence: 2-token forward ≈ 1.23x cost of 1-token; stage boundaries show
0.2-0.26 ms floor per CPU-side sync; small-kernel stages (norms, residual,
router) are ~40% of staged time. KDA layers hide attention behind one fused
kernel already.


## 2026-08-29 — K-width speculative decode (DS4_GLM_MTP_WIDTH) — implemented, default off

Implemented a K-width greedy MTP cycle for GLM-5.3 (`ds4_session_glm_spec_cycle_wide`,
dispatched when `DS4_GLM_MTP_WIDTH > 2`): verifies `first_token` + K-1 nextn
drafts in one layer-major forward, per-row shared-head argmax for accept,
KDA snapshot/restore + prefix replay on reject, chain rebuild from verified
hidden rows. Two non-obvious fixes were required to make deep drafts work:

1. `glm_graph_mtp_step` gained a `raw_hidden` mode: chained drafts consume the
   previous nextn step's own output hidden (snapshot of `g->next` taken BEFORE
   the embed stage overwrites it — the snapshot must precede `enorm`, not sit
   in the hnorm stage).
2. The nextn KV slots at pos..pos+rows-1 must be populated by the previous
   cycle's chain steps; seeding steps are skipped by default
   (`DS4_GLM_MTP_SEED_STEPS=1` restores) because accepted slots are already
   correct and stale slots only hurt draft quality, never output.

Correctness: greedy goldens byte-identical to serial at widths 3, 4, 6
(accept-only-prefix + full re-verify guarantee target-matching output).

Results (M5 Max, n=256, ABBA): K=2 upstream 40.5 t/s; W=3 30.6 t/s; W=4
20.8; W=6 16. Depth-2 draft acceptance is only ~45% even with correct raw
chaining (vs 89.6% depth-1), and each reject costs a second full forward
(KDA restore + prefix replay). Verify is dispatch-bound (~flat to 3 rows),
so the reject tax dominates. **The upstream width-2 path remains the fastest
configuration.** The wide machinery is kept for future work: a KDA state
that supports per-row checkpointing (or MoE-side speculative tricks) would
flip the calculus.

Also tried and rejected: relaxing the `n_expert==6` gate on
`kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16` to include GLM-5.3's 8 experts
(bit-identical output, prefill ABBA median 296 vs 304 t/s — no win at 8/288
fanout, reverted).

## DFlash2 for GLM-5.3 — status

DFlash2 draft GGUFs exist (Anbeeld/incoai, qwen3-arch, 0.43–2.35 GB, same
tokenizer as GLM-5.3). This branch has no dflash2 machinery — it lives in
`origin/ornith15` (`ds4_dflash2.inc`, 1326 lines) bound to the Qwen graph
(`g_qwen_pool`, DSpark spec-frontier integration). Porting it to a
glm5-next target means a qwen3 draft graph inside the GLM session path plus
new draft binding for the GLM spec cycle — a separate feature branch, not a
patch. The in-GGUF nextn MTP (width 2, 89.6% acceptance, 40.5 t/s decode vs
33 serial) remains the draft mechanism for this model.

### Prefill chunk sweep (4500-tok prompt)
DS4_GLM53_PREFILL_CHUNK env added (default 2048). 1024/2048/4096 ABBA:
451/437/449 median — noise-level; default unchanged.
