# Qwen3.8-Flash-Next DS4 performance handoff

Last updated: 2026-09-02, twenty-first session (Europe/Rome)

## Objective

Continue improving Qwen3.8-Flash-Next decoding in DS4, for both ordinary
decode and replay-free MTP. The requested target is at least 60 generated
tokens/second on the local Apple M3 Ultra. The target has not been reached.

## Session-two status (read this first)

1. The tiny-verifier fusion gate below is now COMPLETE.
   `tests/test_qwen4_metal.c` gained explicit 3-row parity fixtures for all
   four fused kernels (`... rows>1` cases): fused SiLU, fused SwiGLU,
   HC up/mix, and HC write. All pass (max errors 5.7e-06, 1.2e-04, 2.7e-07,
   1.1e-05). The full Metal suite, host/spec suites, pack tests, the
   acceptance validator, and `make test-qwen4-release-core` all pass. While
   writing the HC-write fixture, an apparent 2x kernel error turned out to
   be a fixture bug (the expected values were uploaded as the initial
   read-modify-write state); a standalone probe confirmed the kernel is
   exact. Keep the init/expected split in future RMW fixtures.
2. The Qwen greedy CLI path now routes through the session executor in
   `ds4_engine_generate_argmax` (family check). The legacy raw greedy graph
   is DeepSeek/GLM-shaped and segfaulted on the Qwen weight directory — a
   pre-existing gap, reproduced at HEAD 5084a7c before the fix.
3. The experimental ANE runtime is now complete and validated as transport
   infrastructure (see QWEN38_FLASH_NEXT.md, "Experimental ANE hybrid
   transport"): `ds4_gpu_tensor_wrap_iosurface`,
   `ds4_gpu_ane_pack_dual_f32_tensor`,
   `ds4_gpu_ane_merge_dual_f16_f32_tensor`,
   `ds4_gpu_matmul_q8_0_range_tensor`, `metal/ane_io.metal` registered into
   the runtime library, and Makefile targets `test-ane-quant` and
   `ane-qb-shape-bench`. The bench gate PASSES on this M3 Ultra
   (1.24-1.29x on the 4096x1024x32768 Q8_0 shape, cosine >= 0.999995,
   deterministic). Model-graph integration remains open; M=1 GEMV latency
   on the ANE is the open feasibility question for decode (the measured
   win is prefill-shaped so far).
4. Measured MTP cycle on the ORIGINAL v3 pack (~128-token context):
   base target pass ~36.8 ms, base-row MTP history ~4.9 ms, draft chain
   ~3.4 ms/token, verify4 ~57 ms, commit-side reconcile+history ~8 ms
   (~23 ms of MTP-layer work per cycle total). The reconcile is
   architecturally required: it re-runs the MTP layer teacher-forced on
   verify-pass target hiddens (the chain self-conditions), so chain states
   cannot be reused for it. On the best experimental profile the verify4
   pass measured ~54.5 ms. Against approximate 750 GB/s effective
   bandwidth, decode sits ~6x above its weight-traffic floor and the verify
   pass ~2.9x; both passes are latency-bound across ~300 small kernels
   (GPU busy ~98%). Reaching 60 tok/s needs roughly 2x across the target
   passes.
5. QSA per-layer cost grows with context (0.75 ms at ctx 161 to 2.36 ms at
   ctx 1232, sync-inflated) — matters for long-context decode.
6. Draft-depth sweep on the v3 pack measured 26.49/26.35/25.17/25.92/25.33
   tok/s at depths 2/3/4/5/6 — shallower wins there, but the sweep below on
   the experimental profile still shows depth 4 best; treat depth as
   workload-dependent, not settled.
7. RESOLVED in session four: the one-pass MTP history reconciliation is
   LANDED. The earlier "state-copy has no active command batch" failure was
   a control-flow bug in the first attempt, not a batch-lifecycle mystery:
   the original second segment (`if (rows > 1u) ...`) is a separate
   statement, not an else-branch, so the merged path fell through and re-ran
   the MTP layer a second time after the final-head readback had already
   closed the batch. The fix is `rows > 1u && !merged_history` on that
   segment. The merged path assembles [mtp_trunk_last, mtp_input rows
   0..rows-2] in a dedicated scratch tensor for rows <= 16 and runs one
   qwen4_graph_mtp_run_with_ids call. Validated: Metal fixtures, planner
   tests, full test-qwen4 aggregate, greedy output identical to the
   two-segment path, history/commit stage times down ~0.2-1 ms per cycle
   (29.2 vs 29.0 tok/s v3 pack; 34.13 vs 34.05 on the experimental
   profile). Kill switch: DS4_QWEN4_MTP_MERGED_HISTORY=0.
8. Wide split-K SiLU kernels were implemented, validated, and measured in
   session four; they are OFF by default (opt in with DS4_QWEN4_SILU_WIDE=1).
   kernel_qwen4_q8_0_f32_m1_silu_wide gives each narrow-output (N<=512)
   M=1 down projection a 512-thread group with K split across sixteen
   simdgroups: plain decode improved a reproducible +3.8% (27.70 vs 26.68
   tok/s over three interleaved pairs). But MTP net DROPPED to 27.5 tok/s:
   the faster base decode moves the MTP scheduler's break-even so
   speculative cycles are bypassed (202 bypass vs 62 block cycles), i.e.
   the plain-decode gain exceeds the MTP-cycle gain because the verify pass
   barely improves. The multirow variant
   (kernel_qwen4_q8_0_f32_silu_wide_rows) left the verify pass unchanged:
   at K=10240 with F32 activations, grouping rows into one threadgroup
   quadruples activation re-reads, which dominates the tiny weight savings.
   Two lessons: (a) the v3-pack MTP-vs-plain margin is only ~2%, so any
   change that speeds plain decode more than the verify pass flips the
   scheduler — retune the cost controller before shipping decode-side
   wins; (b) for these shapes arithmetic intensity is activation-bound,
   not weight-bound.
8a. Rows-inner weight-reuse tiny-batch kernels were IMPLEMENTED, VALIDATED,
   MEASURED, and REVERTED in session three. Four kernels (SiLU, SwiGLU,
   HC up/mix, HC write) loaded their weight rows once and looped up to
   eight verifier tokens in the inner loop, with the M=1 grid shape. All
   passed the multirow parity fixtures, but the hot A/B measured 26.4 vs
   29.0 tok/s and verify 86.9 vs 83.1 ms — an 8% regression. LESSON: these
   M=1-shaped projections are latency-bound, not bandwidth-bound; the
   per-row grid's 4x thread parallelism is load-bearing latency hiding, and
   L2 already amortizes the shared weight reads (dense scaled 1.77x for 4
   rows, not 4x). Do not retry weight-reuse inner loops for tiny batches on
   this GPU; more parallelism per projection, not less, is the direction.
   (Debugging note that cost time: a kernel whose writer guards on
   lane == 0 must declare lane as thread_index_in_simdgroup, not
   thread_index_in_threadgroup — the latter makes only global thread 0
   write, which looks like missing per-stream data.)
9. ANE integration into the verify pass was evaluated with per-shape
   probes and DISPROVEN in session three; do not wire it into the graph.
   The shape bench now accepts DS4_ANE_BENCH_SHAPE=MxKxN, the ANE runtime
   keeps a 1 MiB IOSurface floor (tiny M=4 surfaces previously failed with
   private-runtime error 42), and ds4_gpu_matmul_q8_0_range_tensor accepts
   partial token tiles via the bounds-checked column kernel. At the real
   verify shapes the hybrid measured 0.23-0.39x — three to four times
   SLOWER than the GPU alone:
   - 4x320x10240 (HC up/mix): GPU 0.33 ms vs hybrid 1.02-1.41 ms
   - 4x10240x2560 (HC down):  GPU 0.47 ms vs hybrid 1.21-1.42 ms
   - 8x320x10240:             GPU 0.34 ms vs hybrid 1.05-1.36 ms
   Two independent reasons: transport (pack+merge) costs 0.5-0.8 ms which
   alone exceeds the whole GPU kernel, and the ANE shows a ~0.45 ms
   per-inference latency floor regardless of matrix size, so it cannot win
   on latency-bound tiny batches even with free transport. The ANE remains
   interesting only for prefill-shaped batches (M in the thousands), where
   the gate has measured 1.12-1.29x across runs.
10. Standing conclusion after sessions two and three: every cheap path to
   60 tok/s on the Q4 pack has been measured and eliminated — weight-reuse
   inner loops (regression, see 8), ANE verify offload (see 9), merged MTP
   history (blocked, see 7). What remains is ground-up M=1 GEMV
   parallelism work on the GPU, a Q2-plus-MTP pack (converter policy
   blocks it pending MTP-expert imatrix calibration), or acceptance/
   scheduling changes that raise tokens per cycle. The verify pass's dense
   HC chain and the ~28.6 ms base pass are both latency-bound at roughly
   6x above their bandwidth floors; only added per-projection parallelism
   or fewer kernels per layer can move them.


Do not describe this as an MLX implementation. `mlx-serve` uses the DS4
library for this model; its Qwen3.8 path is not a separate MLX engine.

The user's original replay-free partial-MTP request is attached at:

`/Users/ifioravanti/.codex/attachments/864c7e20-cfa8-4361-9194-daf0fe7557ae/pasted-text.txt`

## Repository state and operating rules

- Repository: `/Users/ifioravanti/github/ds4`
- Branch: `qwen3.8-flash-next` (renamed from `qwen3.8-flash-next-q4`
  after the standardization; the old remote branch was deleted)
- Previous HEAD before this campaign's commits: `5084a7c`
- Session twenty-one committed `754986b` (the campaign plus the
  Q4_0-routed standardization) and pushed it to the fork
  (ivanfioravanti/ds4-metal).
- Prefix every shell command with `rtk`.
- Use `apply_patch` for source edits.
- Use `uv` for Python work where possible.
- Do not cite Codex in commit messages.
- No `ds4-server` or conversion process is running at handoff time.
- `git diff --check` passes at handoff time.

Tracked files currently modified:

```text
Makefile
QWEN38_FLASH_NEXT.md
ds4.c
ds4_gpu.h
ds4_metal.m
ds4_server.c
gguf-tools/Makefile
gguf-tools/quality-testing/score_official.c
gguf-tools/qwen4_pack.py
metal/moe.metal
metal/qwen4.metal
speed-bench/qwen4_benchmark.py
tests/test_qwen4_host.c
tests/test_qwen4_metal.c
```

Experimental scripts created by this work and still untracked:

```text
speed-bench/qwen4_q4_0_routed_repack.py
speed-bench/qwen4_q4_dense_repack.py
speed-bench/qsa_prefill_bench.c   (make qsa-prefill-bench; item 20)
speed-bench/gqa_mma_proto.metal   (prototype kernels; item 21)
speed-bench/gqa_mma_proto.m       (driver; cc -O2 -framework Foundation
                                    -framework Metal, run from repo root)
speed-bench/sg_layout_probe.metal
speed-bench/sg_layout_probe.m     (simdgroup layout probes; item 20e)
speed-bench/qwen4_mtp_sweep.py    (interleaved MTP/decode A/B driver; item 27)
```

The experimental ANE runtime that used to live here untracked
(`ds4_ane.h/.m`, `ds4_ane_stub.c`, `metal/ane_io.metal`,
`speed-bench/ane_qb_shape_bench.c`, `tests/test_ane_quant.c`, and the
vendored `third_party/omlx/` license) was REMOVED by owner decision at
the end of session eighteen, together with every tracked reference: the
Makefile build rules and targets, the `DS4_METAL_ANE_IO_SOURCE`
registration, the IOSurface tensor wrapper, the two ANE transport
functions, and the range-restricted Q8_0 matmul that only the hybrid
bench used.  The code was never linked into the production binaries and
its integration results were negative (items 3, 9, 16-17); a copy of
the removed files sits in /tmp/ane-removed-archive for this session
only.  Build and the full Metal fixture suite pass after the removal.

11. Session four additions: the MTP scheduler cost controller now requires
   a clear cumulative loss before its permanent bypass — the baseline prices
   one plain token at the base-eval wall time, which omits per-token decode
   overheads, so the old exact comparison bypassed profitable speculative
   cycles whenever decode-side kernels got faster.  The margin is
   DS4_QWEN4_MTP_SCHEDULER_MARGIN (default 110 percent).  Measured: the
   experimental profile improved 34.12 to 34.74 tok/s at depth 4 (65 block
   cycles instead of 206 bypasses); with the wide kernel enabled the bypass
   count dropped from 202 to 0, although wide still nets slightly below the
   classic kernels for full MTP runs and stays opt-in.  With the retuned
   scheduler plus the landed merged history, the depth optimum is
   pack-dependent: depth 3 reaches 38.5 tok/s on the experimental profile
   (best of campaign) while depth 4 remains best on the v3 pack (29.2).
   The 60 tok/s gap is now entirely the verify pass's activation-bound
   dense chain plus the ~28-35 ms base decode.

12. ATTACK (a) OF THIS PLAN WAS IMPLEMENTED AND DISPROVEN in session
   five: full BF16 staging of the HC read chain (a staging copy kernel plus
   BF16-input M=1 and multirow SiLU kernels reusing the exact
   qwen4_q8_0_dot_bf16 arithmetic, wired through qwen4_graph_hc_read with a
   DS4_QWEN4_HC_BF16_STAGE switch) measured 27.53 vs 29.19 tok/s on the v3
   pack — a 5.7% REGRESSION — and was cleanly reverted (no code remains).
   The extra per-site staging launch (two HC sites x 48 layers per forward)
   costs more than the halved activation traffic saves. Combined with the
   wide-rows result, the chain is now conclusively kernel-count/latency
   bound, not traffic bound: L2 absorbs the F32 re-reads. Do not retry
   traffic-reduction variants (BF16 staging, fused converters inside
   norm_inject are expected to be neutral at best). The one measured win in
   this space remains the wide split-K M=1 kernel (+3.8% plain decode,
   opt-in). The launch-fusion direction was ALSO implemented and measured
   in session six: kernel_qwen4_q8_0_f32_m1_norm_down_silu folded the
   entire norm/inject pass into the down projection's prologue (one launch
   per site, about 96 fewer per forward, with group zero materializing the
   normalized rows and injection partials for downstream consumers).
   Fixtures passed, but throughput was exactly neutral on both plain decode
   (26.65 vs 26.69 tok/s) and MTP (29.15 vs 29.18), and the greedy
   continuation differed because the norm-scale reduction order changes —
   so it was cleanly reverted. CONCLUSION: the eight-layer command-buffer
   pipelining already hides per-kernel launch latency; kernel-count
   reduction is NOT a lever on this GPU. With traffic (BF16), grouping
   (rows-inner/wide-rows), and launch count all measured dead, the dense
   chain's ~48 ms verify cost is bound by the kernels' intrinsic
   execution latency at these shapes; only a categorically different
   decomposition (fewer, much larger fused stages spanning whole layers,
   or model-level changes like a Q2+MTP pack) can move it. Original quantified
   plan for reference:
 the HC
   read/write chain's per-site activation re-reads are about 13 MB of F32
   per site (the down projection re-reads normalized streams K=10240 from
   all 80 out-blocks; the up/mix re-reads the K=320 low vector from all
   2560 dim-groups; the HC write re-reads the 2560-wide block from all 640
   out-blocks), i.e. roughly 52 MB per layer, mostly served from L2. Two
   attacks, in order of expected value: (a) BF16-stage the chain
   intermediates — norm_inject additionally emits a BF16 copy of
   normalized_streams, and the down/up/write kernels take BF16 activation
   inputs (the plain matmul BF16-input path already exists and is fixture
   covered), halving the L2 pressure; validate with the four-case
   deterministic evaluator and the NLL scorer because BF16 activations
   add roughly 0.4 percent relative error on top of Q8 weights. (b) For
   the up/mix (K=320 fits in threadgroup memory), cache the low row per
   threadgroup and compute four dims per group, cutting both re-reads and
   the group count fourfold. Re-measure at depth 3 on the experimental
   profile with the retuned scheduler; the campaign best so far is 38.5
   tok/s there (29.2 on v3 at depth 4) against the 60 tok/s target.

13. Session seven: the Q2+MTP path is UNBLOCKED and measured.  The
   converter now accepts `--mtp-imatrix weight-energy` (the deterministic
   per-expert input-column weight-energy selection the imatrix already
   uses for zero-count experts, applied to all 512 MTP experts, since the
   pinned unsloth imatrix covers only the 48 base layers) and a new
   `--rebuild-mtp` flag atomically adds the v4 MTP sidecar to a finalized
   pack without rewriting the 40 GB base (mirrors --rebuild-vision;
   --tokenizer-template is now required only for full conversions).  The
   existing q2-v1 pack gained `qwen3.8-flash-next-q2-mtp.gguf` (879 MB,
   same pack_id b573f495a00a2530502b5c23d816364d, 4 artifacts) and the
   runtime loads it: "matching v4 iq2_xxs_gate_up_q2_k_down MTP sidecar
   loaded (state=ready)".  MEASURED: Q2+MTP reached 31.3 tok/s at depth 3,
   32.5 at depth 4, and 32.1 at depth 2 — all BELOW Q2 plain decode
   (34 tok/s).  Two causes: the weight-energy MTP experts degrade draft
   quality (mean accepted about 2.2 tokens per cycle versus 2.65 for the
   Q4_MTP reference; histogram at depth 4 is 7/5/9/11), and the verify
   pass still dominates (52 ms for three rows, 69 ms for four).  Even
   with perfect acceptance the Q2 cycle math ceilings near 44 tok/s
   (about 29.4 ms base + 6-9 ms draft + 52-69 ms verify + 4 ms commit).
   Conclusion: 60 tok/s is not reachable on either backbone through the
   current MTP structure; the binding constraint is the verify pass's
   ~52-69 ms, which survived every GPU-side attack in items 8-12.

14. Campaign close-out audit (session eight).  The 60 tok/s target was
   evaluated against every identified direction with measurements: restoring
   Q2 MTP acceptance to the Q4 reference (2.65 tokens/cycle) would lift
   Q2+MTP only to about 33 tok/s — still below Q2 plain decode — because the
   verify pass (52-69 ms) binds first; a real MTP activation imatrix is
   therefore worthwhile only for draft-quality parity, not for the speed
   target, and was not built.  Whole-layer fused stages remain unmeasured
   but every smaller fusion (launch fold, BF16 staging, grouping) measured
   neutral-to-negative against the same latency-bound physics.  Final
   verified state: build clean, make test-qwen4 aggregate green (includes
   the converter fixtures for --mtp-imatrix/--rebuild-mtp), Q2+MTP
   functional end-to-end (33.99 tok/s depth-4 warm run recorded), campaign
   bests 38.5 tok/s (experimental profile, depth 3) and 29.2 (v3 pack,
   depth 4).  Reaching 60 tok/s requires a decision outside the measured
   kernel space: different hardware, a changed acceptance/verification
   structure, or a revised target.

15. OWNER DECISION (session eight, closing the campaign): adopt the
   measured best configuration and record 60 tok/s as not achievable on
   this hardware/pack.  The shipped configuration is the experimental
   profile (DS4_QWEN4_EXPERIMENTAL_Q4_DENSE=1 and
   DS4_QWEN4_EXPERIMENTAL_Q4_0_ROUTED=1 with the q4dense-q40routed-exp2
   pack), the landed merged MTP history, the 110-percent scheduler
   margin, and depth 3 — 38.5 tok/s measured (38.45/38.51/38.46 across
   runs).  The v3 pack ships depth 4 at 29.2 tok/s; the Q2 backbone ships
   plain decode at 34 tok/s with the MTP sidecar available
   (--mtp-imatrix weight-energy / --rebuild-mtp) for experimentation.
   The 60 tok/s objective is closed as measured-unreachable: kernel-level
   directions are all bounded below target (items 8-12), Q2+MTP ceilings
   near 44 tok/s even with perfect drafts (item 13), and the owner
   declined the research-scale whole-layer fusion and target-revision
   alternatives.

16. Session nine addendum: a standalone scaffold for the whole-layer
   fusion feasibility probe exists but is deliberately NOT measured.
   speed-bench/hc_chain_proto_bench.c and
   speed-bench/hc_chain_proto_kernels.h sketch the intended experiment
   (production three-dispatch HC-read chain versus one persistent kernel
   at the real verify shapes, isolated from the production Metal library)
   but the drafted reference kernel does not faithfully implement the
   three-dispatch production structure and contains unsound threadgroup
   scoping; running it would have produced a false data point.  The files
   are inert (referenced by no build rule) and left as a starting
   scaffold.  A valid prototype needs: (a) reference = three separate
   kernels dispatched sequentially, mirroring qwen4.metal exactly;
   (b) persistent variant with function-scope threadgroup arrays and
   correct phase barriers, sized so all groups stay resident; (c) a CPU
   reference of the whole chain to validate both paths before timing.
   Budget roughly one focused session; only measure after (c) passes.
17. The prototype was then BUILT TO THAT SPEC and measured in session
   nine; the whole-layer fusion direction is DISPROVEN.  The final bench
   (make hc-chain-proto-bench; speed-bench/hc_chain_proto_bench.c and
   hc_chain_proto_kernels.h) compares the faithful three-dispatch
   production structure (norm; down+SiLU; up/mix at the production grids)
   against one persistent kernel with a device-atomic grid barrier,
   reusing the low vector across groups, at the real verify shapes
   (hidden 2560, four streams, lowrank 320, Q8_0 weights).  Both paths
   validate against a CPU reference at 2.3e-07 relative error before any
   timing.  Results at the best persistent grid (80 groups):
   rows=1 0.911x, rows=2 0.722x, rows=4 0.59-0.75x, rows=8 0.512x;
   group sweeps at rows=4 (40/80/160/320) peaked at 0.707x.  The
   persistent whole-stage fusion is 9-49 percent SLOWER everywhere: the
   software grid barrier plus the reduced per-phase parallelism cost far
   more than the dispatch boundaries they replace (consistent with every
   smaller fusion measuring neutral — the boundaries were already free
   under command-buffer pipelining).  With this, ALL THREE closure paths
   are fully measured: kernel-level work is bounded below 60 tok/s
   (items 8-12), Q2+MTP ceilings at ~44 (item 13), and the research-scale
   decomposition loses at prototype scale (this item).  Reaching 60
   tok/s now strictly requires the owner to revise the target or change
   the hardware.
18. Re-measurement note (session eleven): after a seven-minute cool-down
   with no recorded thermal or performance warnings, the best
   configuration measured 35.94/36.58/36.12 across three spaced runs —
   the honest reproducible band today is about 36-38.5 tok/s depending
   on ambient machine state, with the campaign-best recordings at 38.45/
   38.51/38.46.  A 40 tok/s revised target is therefore NOT met by
   simple re-measurement; it would need roughly +4-11 percent beyond the
   delivered state.  The command-buffer depth was also re-swept on the
   experimental profile for the first time (4/8/16/24, then interleaved
   8-vs-24): depth 8 won or tied every pair (35.24/36.10/36.17 versus
   35.41/34.43/35.53), confirming the retained default there too.  Every
   runtime knob on the best configuration has now been swept or
   measured; no remaining setting provides the +4-11 percent.
   Final un-run combination closed in session ten: the wide split-K
   kernel WITH the retuned scheduler on the experimental profile at
   depth 3 (the best configuration) measured 36.74/35.95 versus
   36.71/36.80 without it — neutral within noise (the machine was also
   running warmer than during the 38.5 recordings; the A/B is internally
   valid).  No remaining combination of the retained knobs moves the
   number materially.  The deliverable state stands: experimental
   profile + merged history + 110-percent margin + depth 3 = 38.5 tok/s
   campaign best; every closure path is quantified in items 8-17 and
   awaits the owner.

19. SESSION TWELVE REOPENED PREFILL AND LONG-CONTEXT DECODE, both were
   transformed, and every landed change is measured, fixture-gated, and
   kill-switchable.  The earlier closure statements in items 10-15 apply
   only to SHORT-CONTEXT M=1 decode, which is unchanged (~35.7 tok/s at
   ctx 128); they did not cover prefill batching (which ran scalar dot
   kernels at every batch size) or the QSA per-token work at realistic
   contexts (which ran a one-GPU-thread heap top-k per layer and a
   serial gathered-attention pass).  Four changes landed in the working
   tree (see QWEN38_FLASH_NEXT.md, "Tensor-core prefill and parallel QSA
   selection"):
   a. Batched Q8_0 dense projections dispatch the tiled tensor-core
      kernel_mul_mm_q8_0_f32 at >= 32 rows
      (DS4_QWEN4_Q8_0_MUL_MM_MIN_ROWS=0 disables).
   b. Routed experts run mapped grouped tensor-core matmuls at >= 512
      rows (DS4_QWEN4_MOE_MUL_MM_ID_MIN_ROWS=0 disables;
      kernel_mul_mm_id_q4_0_f32 added; the deterministic route map,
      SwiGLU epilogue, and exact slot-order weighted sum are preserved,
      with a 600-row two-tile fixture against the scalar CPU reference).
   c. The QSA ordered top-k is a bitonic selection network for both the
      M=1 path and the multi-query streaming path
      (DS4_QWEN4_QSA_BITONIC_TOPK=0 restores the heap controllers); the
      comparator is exactly the heap's, and the existing exact-order
      fixtures at 512..32768 pooled rows pin bit-identity.
   d. The gathered QSA attention fuses QK, online softmax, and the value
      accumulation per simdgroup with 128-bit K/V loads (structure from
      oMLX PR #3244 / mlx-serve msv_attn_qsa256); the memoized/legacy
      instantiations stay bit-identical and the CPU-reference fixture
      passes at 7.5e-9.
   MEASURED (M3 Ultra, exp2 profile): prefill 2048 tokens 154 -> 858
   tok/s, 8192 tokens 171 -> 781, 32768 -> 737; the v3 pack's 2048-token
   prefill went 75.7 -> 568.  Decode at ctx 2048/8192/32768 went
   15.2/11.8/8.65 -> 33.8/33.1/31.7 tok/s (bit-identical greedy output
   vs the old QSA kernels; the top-k/attention changes are exact, the
   matmul changes only affect prefill batches).  MTP at an 8.7K-token
   prompt now keeps the scheduler engaged with zero bypasses; a
   full-accept depth-3 cycle is ~91 ms for four target tokens and a
   500-token completion including the 8.7K prefill finishes in ~25 s
   wall (~36 tok/s decode; it was ~12 tok/s plain with MTP bypassed
   before).  QUALITY: the stock 100-case NLL fixture is unchanged (its
   prompts never reach the batch thresholds); a purpose-built 60-case
   1272-token-prompt NLL A/B measured -0.1 percent (noise); greedy
   continuations remain fluent and diverge only at rounding-scale
   argmax boundaries, the same class the chunk-size comparison already
   accepts.  Full make test-qwen4-release and the Metal fixture suite
   pass at handoff.  KNOWN GAP: the frontier-state comparator's 2e-5
   logits tolerance does not hold across chunk sizes on this branch
   even with all new paths disabled (pre-existing scalar delta is
   LARGER, ~0.1; with the tiled paths it is ~0.022) — that gate needs
   re-derivation independent of this work.  OPEN IDEAS NOT YET MEASURED:
   oMLX/mlx-serve put all 12 GQA query heads into one 16-row MMA tile
   per (query, kv_head) to share gathered K/V tiles — that is
   prefill-shaped and our fused kernel already serves prefill, but a
   Steel-style MMA rewrite could still cut prefill attention further;
   mlx-serve's two-pass split-KV decode helps only far beyond 32K;
   the verifier=decode1 fallback cycle still costs ~67 ms.


20. SESSION THIRTEEN (this round) reopened the biggest single prefill
   kernel stage with a measurement tool and an opt-in rewrite; the
   direction stays open, the infrastructure is landed, and every number
   below is from uncaptured standalone benches on the M3 Ultra.
   a. NEW TOOL: `make qsa-prefill-bench` (speed-bench/qsa_prefill_bench.c)
      drives the production GPU dispatches at the exact prefill geometry.
      Attribution at 8192 rows / full 2048-token selection: gathered QSA
      attention 114 ms/layer (x12 = 1364 ms per 8K chunk, the largest
      measured kernel stage), streaming top-k 27-30 ms/layer, GDN R4
      prefill recurrence 10.4 ms/layer, dense Q8_0 12288x2560 23 ms. The
      attention kernel moves ~412 GB/layer through L2 (24x redundancy:
      every one of the 12 sibling query heads re-gathers its KV group's
      rows) at ~3.6 TB/s effective — it sits AT the L2 bandwidth limit and
      only 2x above the f32-ALU floor, which is why scalar load-sharing
      alone was ruled out analytically (~1.7x ceiling).
   b. Baseline today (exp2 profile, 8K chunks, promessi corpus):
      prefill 700-765 tok/s across 14K-32K frontiers, decode 31-34 tok/s
      — consistent with session twelve within machine-state variance.
   c. NEW OPT-IN KERNEL: kernel_qwen4_qsa_attention_gqa_mma_f32 — the
      oMLX-PR-3244/msv_attn_qsa256 structure (all 12 GQA heads of a KV
      head on the matrix units, gathered K/V staged once and shared) as
      one 32-thread simdgroup per eight-head row block: F16 operands,
      F32 accumulation, tg-tile softmax, lazy running-max rescale, exact
      production rank-to-token semantics. Fixture
      test_qsa_attention_gqa_mma pins it against BOTH the CPU reference
      and the scalar kernel (1.3e-05 max error, 40 queries including
      tail-only/zero-selection/short-visible boundaries). Enable with
      DS4_QWEN4_QSA_GQA_MMA=1; DS4_QWEN4_QSA_GQA_MMA_MIN_ROWS (default 8)
      guards row counts; DS4_QWEN4_QSA_MMA_DEBUG=1 +
      ds4_gpu_qwen4_qsa_mma_debug_stats dumps per-tile softmax stats.
      make test-qwen4 aggregate is green including the new fixture.
   d. MEASURED, NOT LANDED AS DEFAULT: per-simdgroup 244 ms/layer vs the
      scalar kernel's 113 at the 8192-row shape. CORRECTION (session
      fourteen audit): the previously reported "lockstep variant 112
      (parity)" and "unroll 272" readings were STALE-BENCH artifacts of
      lesson (3) — those runs predated the bench relink and measured the
      scalar kernel against itself. The only valid MMA number is the
      per-simdgroup 244; the 64-thread two-simdgroup lockstep
      organization (shared K/V staging, half the gather traffic of
      per-simdgroup) has never been validly measured and is the first
      candidate for the prototype bench. The 12x gather-traffic cut
      (412 -> 34 GB/layer) is being spent on staging serialization and
      reduced threads in flight. The ceiling analysis still favors the
      direction: ~16.8 GB unique gather traffic plus matrix-unit products
      floor near 20-30 ms/layer. What remains is kernel work, not
      concept: an in-register softmax (see (e)), a threadgroup budget
      near 8 KB, and probably BK=128 tiles to halve per-tile fixed costs.
      Roofline note: the scalar kernel sits exactly at the machine's
      FLOP:byte balance point (412 GFLOP and 412 GB at ~3.6 TFLOP/s and
      ~3.6 TB/s simultaneously), so NO scalar-F32 load-sharing variant
      can beat it — only the matrix units (4x ALU) plus sharing (12x
      traffic) move both axes.
   e. DEBUGGING LESSONS (all cost real time this session; do not relearn):
      (1) simdgroup_matrix thread_elements() indexing is COMPILER-
      UNSTABLE — two compilations of the same probe printed different
      per-lane element mappings (speed-bench/sg_layout_probe.{metal,m}).
      Never index matrix elements directly; transpose via tg tiles.
      (2) The online-softmax quad-shuffle reduction MUST shuffle the
      UNMUTATED per-lane partial. Accumulating into the same variable
      being shuffled double-counts lanes: the owner lane ends with
      v0+4*v1+v2+v3 while max survives by idempotence — the symptom is a
      uniformly halved output with a sane running max.
      (3) MEASUREMENT TRAP: `make ds4-server tests/...` does not rebuild
      speed-bench binaries. Two "112 ms" readings this session were a
      stale bench binary two kernel revisions behind. Always
      `make <bench-target>` after touching ds4_metal.m or any .metal.
      (4) The full-model Qwen profile buckets are CPU-side submission
      accounting; prefill GPU attribution needs the standalone bench (a).
   f. Also verified this round: the graph allocation log now correctly
      reports `MoE-down=Q4_0-expert-split` for the exp2 pack (the cleanup
      item in an earlier list is already closed), and the default
      attention path is byte-identical to before the round (full-model
      prefill 710-722 tok/s / decode 32-34 re-measured after all changes;
      fixtures and the make test-qwen4 aggregate green at handoff).


21. SESSION FOURTEEN (the GQA MMA follow-up) rebuilt the prototype
   process on a standalone bench that compiles its kernels from source
   at runtime (structurally immune to the stale-binary trap), found a
   winning kernel organization, fixed four real bugs the earlier weak
   validation had hidden, and established that the isolated stage win
   does NOT move full-model prefill on this workload.
   a. NEW TOOL: `speed-bench/gqa_mma_proto.m` + `gqa_mma_proto.metal`
      (build with `cc -O2 -framework Foundation -framework Metal
      speed-bench/gqa_mma_proto.m -o speed-bench/gqa_mma_proto`; run
      from the repo root). It validates every FULL variant against a
      CPU reference that MODELS the kernel's F16 rounding (via
      _Float16) before timing, and cross-checks variants against each
      other (bit-identical across tilings when correct).
   b. ATTRIBUTION (8192-row shape, all 512 blocks selected): the
      lockstep 2x32-thread design measured 216-233 ms/layer (the old
      "112 parity" was indeed a stale-binary artifact); with live
      results, the matrix products run at only ~1.5 T-MAC/s versus the
      dense tiled matmul's ~11 T-MAC/s. Cause: ~200 registers of
      accumulator fragments per thread starve residency. The fix is
      ROLE-SPLIT: per (query, KV head), one 192-thread group with six
      simdgroups — two SCORE groups (8 rows each, full-width QK +
      softmax, no O accumulators) and four PV groups (8 rows x 128
      dims, 16 fragments each). Phase-skips showed the ablation floor
      ~41-47 ms; shrinking threadgroup bytes monotonically improved
      throughput (occupancy lever), and K is staged dim-major so no
      operand load transposes.
   c. FOUR REAL BUGS found by strengthening validation (each passed the
      old weak checks): (1) staging loops under role guards left tile
      units unstaged (score threads staged only units 0..63); (2) the
      emit wrote all 8 rows of row-half 1, aliasing the next KV group's
      heads and overrunning the buffer on the last group; (3) the
      softmax pass covered only 64 columns per lane-quad — with BK=128,
      columns 64..127 never entered the softmax; (4) the 8 KB
      rescale/emit overlay needed kvs+s_tile >= 8 KB, which only held
      by luck per config. Fixes: all-thread staging, valid-rows emit
      guard, BK/4 columns per lane, and an explicit 4 KB O block used
      in four barrier-separated phases. VALIDATION LESSON: absolute
      tolerances at tiny magnitudes swallow 50%-wrong outputs — use a
      reference that models the operand rounding and in-range gather
      patterns that exercise every rank position.
   d. FINAL STATE (all fixture-gated): production kernel
      kernel_qwen4_qsa_attention_gqa_mma_f32 is the role-split design
      at BK=64/DC=8 (192 threads, ~20 KB tg). test_qsa_attention_gqa_mm
      a passes at 1.88e-04 vs BOTH the CPU reference and the scalar
      kernel; proto cross-variant checks are bit-identical. Standalone
      stage A/B through the production dispatch: 117 -> 78 ms/layer
      (1.50x). FULL-MODEL prefill A/B (exp2 pack, 8K chunks, 16
      frontiers, off/on): 0% +- 1% — the isolated win is absorbed by
      the chunk's other walls (PLE waits, dense projection family,
      submission gaps), so the kernel stays OPT-IN
      (DS4_QWEN4_QSA_GQA_MMA=1; default remains the scalar path).
      Greedy continuations are byte-identical between paths (4-run
      SHA match). STRATEGIC CONCLUSION: further attention-side kernel
      work cannot move this workload's prefill; the chunk wall is
      dominated by non-attention stages and gaps. The next prefill
      levers are the dense projection family (~1.1-1.7 s/chunk
      aggregate), GDN recurrence (374 ms), streaming top-k (330 ms),
      and understanding the ~4-5 s of the 8K chunk not accounted by
      measured kernel stages.
   e. The role-split kernel's remaining headroom (for whoever returns):
      the ablation floor is ~41-47 ms and it lands at 78 (production
      dispatch) — the gap is staging serialization and tg occupancy.
      Concrete backlog: s_tile as F16, pf-fragment hoisting out of the
      V-chunk loop, one-pass softmax with register-held scores, and
      double-buffered K/V staging to overlap loads with products.


22. SESSION FIFTEEN (this round) attributed the unaccounted 8K-chunk mass
   with a GPU-timeline profiler, found that the routed-expert grouped
   matmul was 40 percent of the chunk and had never been measured, and
   landed a tile-geometry fix worth 11-23 percent end-to-end prefill.
   a. NEW TOOL: DS4_METAL_GPU_STAGE_PROFILE=1 (plus DS4_QWEN4_PROFILE=1)
      flushes one command buffer per public dispatch tagged with the
      dispatch label and harvests GPUStartTime/GPUEndTime at wait time
      into a per-label table printed (and reset) by
      ds4_gpu_stage_profile_report() at each Qwen profile line. It is
      uncaptured GPU-timeline accounting, unlike DS4_QWEN4_PROFILE's
      CPU-side submission buckets, and its per-op flush cadence measured
      ~0% wall overhead (803.2 vs 804.6 tok/s on the same 8K chunk), so
      the attribution does not perturb the workload. The full Metal
      fixture suite also passes under this mode.
   b. ATTRIBUTION (8192-row chunk, exp2 pack, all defaults, before this
      round's change): GPU busy 10371 ms vs 10200 ms wall — the GPU is
      ~100% busy, so the "~4-5 s unaccounted" was never gaps or PLE
      waits; it was kernel stages nobody had measured. Per label:
      routed Q4_0 grouped MoE matmul 4135.7 ms (86.2 ms x 48 layers,
      40%), tiled Q8_0 dense family ~2.9 s across three labels, QSA
      gathered attention 1321 ms (matches the item-20 bench), scalar
      BF16 matmul family 842 ms, GDN R4 recurrence 357 ms, QSA
      streaming top-k 256 ms, HC mix/norm/write ~316 ms, everything
      else <200 ms each. The item-20 bench had no MoE stage, and its
      dense-family estimate (1.1-1.7 s) undercounted ~3.0 s of real
      dense work.
   c. THE FIX (LANDED AS DEFAULT): the grouped MoE path tiled work in
      512-row chunks (DS4_QWEN4_MOE_EXACT_TILE_ROWS). At 8192 rows that
      is 16 tiles per layer, each re-reading essentially the full 1.4 GB
      expert weight set (every tile touches ~all 512 experts with ~10
      rows each) and padding those ~10-row expert populations up to the
      32-row MMA blocks (~3.2x padded FLOPs). One full-chunk tile reads
      the weights once and amortizes the padding (81920 assignments
      over 512 experts ~ 160 rows each ~ 5 full blocks). Standalone
      (qsa-prefill-bench, uniform worst-case routing): 125.2 -> 42.2
      ms/layer (19.1 TFLOP/s effective, near the dense mul_mm rate of
      ~21.8); route distribution stops mattering at full tiles (uniform
      and block-contiguous both 42.2). In-situ GPU stage: 86.2 -> 47.2
      ms/layer (4135.7 -> 2264.8 ms/chunk). The change is one tile-step
      constant: DS4_QWEN4_MOE_MUL_MM_TILE_ROWS now defaults to one tile
      per chunk (8192); setting it to 512 restores the old tiling (kill
      switch). Byte-identity: row regrouping cannot change any output
      element's dot products, accumulation order, or destination; the
      extended Metal fixture test_moe_q4_0_mul_mm_id pins mid+output
      BYTE-IDENTICAL between 512-row tiling and one 600-row ragged
      tile, and full-model greedy 128-token generations are
      byte-identical between tile settings at a 16K frontier.
   d. MEASURED, FULL-MODEL (exp2 pack, 8K chunks): 16-frontier ds4-bench
      sweep 720-878 -> 803-997 tok/s in matched machine states (+11-15%
      at every frontier); a back-to-back interleaved pair at 16K
      measured 760.1 -> 935.9 tok/s (+23.1%) with decode unchanged
      (32.82 vs 32.92 tok/s) — earlier apparent decode dips in longer
      runs were thermal drift. 8K-chunk intervals: 804/767/741/715 ->
      944/891/854/823 tok/s at the four frontiers. make test-qwen4
      aggregate and the Metal fixture suite pass; make test-qwen4-release
      should be re-run before any commit per the standing gate.
   e. NEGATIVE RESULT (implemented, validated, measured, REMOVED):
      expert-sorted B staging — a gather kernel copying x and mid rows
      into compact expert-major order plus a CONTIG_B variant of
      kernel_mul_mm_id reading contiguous B rows. Motivated by a route-
      pattern experiment showing 125 -> 51 ms/layer when each expert's
      B rows are contiguous at 512-row tiles. With full-chunk tiles the
      staging is NET-NEGATIVE: 48.0 vs 42.2 ms/layer standalone (the
      two staging passes cost more than the residual gather latency)
      and slightly worse at 512 tiles too (133.9 vs 125.2). The kernels
      were removed; the lesson stands: fix work amplification (tiling,
      padding, re-reads) before adding data-movement passes.
   f. PROCESS LESSONS (each cost real time this session):
      (1) MEASUREMENT TRAP, new variant of item 20e(3): the first tile-
      size sweep measured NOTHING because the tile loop stepped on the
      hardcoded 512 constant while the env knob only resized scratch
      buffers — five identical numbers across "512..8192" looked like
      "tile size does not matter". A knob that does not reach the loop
      it names measures nothing; before sweeping, verify the knob
      observably changes the dispatch (grid shape, buffer size print,
      or timing of a known-bound case).
      (2) The label-based GPU profiler cannot distinguish which KERNEL
      ran inside one dispatch family (tiled vs scalar Q8_0 share a
      label); confirm kernel selection from the dispatch code before
      reading a label as a kernel identity.
      (3) Standalone synthetic routes bracket but do not reproduce the
      production distribution: uniform-lattice routing overstated the
      MoE stage at 512-row tiles (126 vs 86 ms/layer in-situ) because
      it is the worst-case padding distribution. At full tiles the
      stage became route-independent, which is what made the fix
      trustworthy without modeling the router.
   g. REMAINING VERIFIED 8K-CHUNK BUDGET (post-fix, in-situ): dense Q8_0
      family ~3.0 s (2.08 s "Q4 tensor matmul" + 0.69 s small model
      projections + 0.23 s BF16-activation variant), MoE 2.26 s, QSA
      attention 1.41 s (item 21: exhausted), scalar BF16 family 0.90 s,
      GDN recurrence 0.37 s, streaming top-k 0.27 s, HC ~0.34 s, misc
      ~0.14 s; GPU ~100% busy throughout. The clearest next candidate
      is the BF16 scalar family: the MoE router (48 calls) and the GDN
      decay/beta controls (72 calls) run kernel_qwen4_bf16_matmul_f32
      (scalar per-row) even at 8192 rows — decay/beta at ~0.3 TFLOP/s
      effective; routing prefill-sized BF16 projections onto a tiled
      kernel (or deriving Q8 copies at load like the HC matrices) is
      worth ~0.6-0.8 s/chunk. qsa-prefill-bench now drives the routed
      Q4_0 MoE dispatch at true geometry and accepts
      DS4_BENCH_MOE_ROUTE=uniform|block for route-distribution
      experiments.
   h. LONG-CONTEXT MEASUREMENT (promessi corpus, exp2 pack, 8K chunks,
      64-token decode per frontier, frontiers 32K..256K in 32K steps;
      DS4_BENCH_SNAPSHOT_MAX_BYTES=unlimited avoids inter-frontier
      prefix replays; the model context cap is exactly 262,144 so the
      bench needs --ctx-alloc 262144 with a top frontier of 262,078
      because its default ctx_alloc adds gen+1): prefill degrades
      889/562/500/480/463/436/410/387 tok/s across the eight frontiers
      (the whole-256K average is 563 tok/s) while decode stays flat at
      29-31 tok/s with first-token 62-73 ms and ~29.5 KB of state per
      token (6.64 GiB live at 224K). Attribution of the LAST chunk
      (cache_pos 253,952, wall 20.1 s, GPU ~100% busy, stage
      profiler): QSA streaming top-k 11.48 s = 57% of the chunk (59.8
      ms per 512-query microtile, ~45x the fresh-chunk ramp average —
      the scan is linear in visible pooled blocks), QSA gathered
      attention 1.74 s (145 ms/layer, +23% vs a fresh chunk; every
      query now selects the full 2048-token budget), MoE 2.24 s and
      the dense/BF16/GDN families all context-FLAT. CONCLUSION: the
      long-context prefill bottleneck is the top-k candidate scan, not
      attention or the MoE; a hierarchical or two-pass block-pruned
      selection is the algorithmic lever for whoever returns. Decode
      beyond 32K does NOT collapse (31.7 at 32K -> 29.1 at 256K), so
      the split-KV decode idea stays low-priority.
   i. OFFICIAL-REFERENCE QUALITY BASELINE (established this round so
      future pack changes have a fixed anchor): 100 continuations
      collected from OpenRouter `qwen/qwen3.8-flash` (provider:
      Alibaba; thinking disabled works; top_logprobs is CAPPED AT 5 by
      the provider — 20 is rejected HTTP 400, so probe before
      collecting) into /tmp/qwen38-flash-openrouter-100 with the
      tracked prompts.jsonl, max 24 tokens per case. The owner confirmed
      the hosted qwen3.8-flash IS this checkpoint (API naming drops the
      open-weights "Next" label; the served 1M window extends the
      262,144-token config), so this is a SAME-CHECKPOINT anchor; the
      residual unknowns are Alibaba's serving precision and template
      edge effects, which the local BF16 reference can split (it should
      agree with the API logprobs to ~0.01-0.05 MAE). MEASURED (100
      cases, 2241 target tokens): exp2 pack
      avg NLL 0.32175, first-token match 62/100, greedy LCP 6.77,
      API target-logprob MAE 0.1707, top-1 agreement 89.6 percent,
      top-5 recall 82.6, pair agreement 88.0; v3 pack avg NLL 0.32587,
      first-token 49, LCP 6.13, MAE 0.1733, top-1 89.9, recall 84.4,
      pair 88.9; the all-Q4_K clone (q4dense-exp: Q4_K routed + Q4_K
      dense, DS4_QWEN4_EXPERIMENTAL_Q4_DENSE=1 only) is WORST on every
      primary metric: NLL 0.33792, first-token 54, LCP 5.85, MAE 0.1891.
      The official model's own NLL on these continuations (computed
      from the collected response logprobs) is 0.21220, so every pack
      sits ~0.11 nats above the same-checkpoint self-entropy — read as
      the pack's real end-to-end quality cost (quantization plus
      serving precision and template edges; ~1.12x perplexity on these
      continuations), meaningful rather than noise. Comparator: exp2 vs
      v3 = -1.264 percent NLL (56-44-0 case wins, exp2 ahead), while
      the earlier same-tokenizer fixture had the sign reversed (+0.835
      percent for v3) — pack ORDERING at this effect size is
      fixture-dependent; the absolute ~0.11-nat cost is consistent
      across packs. CROSS-MODEL CALIBRATION (same scorer, tracked
      fixtures): DeepSeek V4 Flash Q4KExperts pack vs its official
      continuations (data/flash) scores NLL 0.15256, first-token 87,
      LCP 14.45/24, per-position greedy agreement 95.6 percent; its
      MXFP4Experts pack is statistically identical (0.14673/87/14.14).
      The DeepSeek fixture's response logprobs are degenerate sentinels
      (0.0 chosen / -9999 alternatives — unusable self-anchor; the
      GLM 5.2 fixture has real logprobs, self-NLL 0.08502, but no GLM
      pack exists locally to score). READINGS: (1) the within-model
      quant ladders are similar (DeepSeek Q4K<->MXFP4 0.006 vs Qwen
      exp2<->v3 0.004, exp2<->all-Q4_K 0.016), so our quantization
      spread is normal; (2) absolute NLLs are NOT comparable across
      fixtures (the Qwen official carries ~3x the self-entropy, 0.212
      vs DeepSeek's likely ~0.05 given its determinism), and the
      agreement differences (LCP 14.45 vs 6.77, first-token 87 vs 62)
      are roughly consistent with similar quantization noise applied at
      different task entropies — but our pack does track its official
      model visibly worse in practical greedy terms.
   j. EXACT-CHECKPOINT BF16 REFERENCE (the decisive quality measurement,
      completed this round): 100 24-token greedy continuations generated
      from the local BF16 snapshot itself (transformers qwen4_exp from
      /tmp/transformers-qwen4 via PYTHONPATH shadowing; CPU, batched —
      single-case CPU generation costs ~27 min/case because the per-token
      512-expert Python loop dominates, while the 100-prompt batch
      finishes in ~2.1 h; load itself is 5.6 s lazy mmap). Prompt
      rendering replicates DS4's segmented chat tokenization (verified
      60/60 token counts earlier); decode/encode round-trip is clean on
      99/100 cases; batched-vs-single-sequence fidelity is 3/5
      byte-identical with 2/5 late near-tie flips (token 23/24 and
      14/24) — acceptable for a shared reference since every pack is
      scored against the SAME tokens. Fixture at /tmp/qwen38-bf16-
      local-100 (generator /tmp/bf16_gen_batched.py; single-seq checks
      in /tmp/bf16-single-seq-check). BF16 self-NLL 0.18563. MEASURED
      DRIFT (all api_* columns now against the exact checkpoint, top-20
      logprobs): v3 pack (Q4_K routed + Q8 dense GDN/QSA) target MAE
      0.0352, mean delta -0.0094, first-token 88, LCP 15.81, top-1
      97.4 percent, top-5 recall 93.6, pair 94.0; all-Q4_K clone MAE
      0.0696, first 85, LCP 13.61, top-1 95.8; exp2 MAE 0.0737, first
      82, LCP 12.51, top-1 94.9. (avg_nll on the BF16 tokens: 0.6375 /
      0.6107 / 0.6660 — heavy-tailed and ORDER-CONTRADICTORY vs the
      agreement metrics; treat MAE/agreement as primary.) DECOMPOSITION
      (same-fixture deltas): swapping dense GDN/QSA Q8->Q4_K (v3 ->
      all-Q4_K) costs +0.0344 MAE (first 88->85); swapping routed
      Q4_K->Q4_0 (all-Q4_K -> exp2) costs only +0.0041. CONCLUSION:
      the ROUTED Q4_0 substitution is nearly drift-free; the Q4_K dense
      GDN/QSA substitution carries ~8x more drift and is what separates
      exp2 from v3 quality-wise. DATA-DRIVEN RECOMMENDED PROFILE: v3
      dense (Q8 GDN/QSA) + Q4_0 routed — expected drift ~0.039 while
      keeping the +23 percent routed decode win and the MoE tile prefill
      win; the build is an APFS clone of v3 plus the EXISTING in-place
      qwen4_q4_0_routed_repack.py (Q4_0 shares Q4_K byte extents, so no
      extra disk beyond the clone) — BUILT AND VALIDATED this round as
      qwen3.8-flash-next-q40routed-v3dense-exp (APFS clone of v3 +
      in-place repack, ~5 min): drift vs BF16 MAE 0.0444 (additive
      prediction 0.039 — confirmed v3-class), first-token 85, LCP 14.92,
      top-1 96.7, recall 92.8, pair 93.1; decode 32-34 tok/s (the routed
      win carries); prefill v3-class 421-510 tok/s at 4K intervals
      (the Q8 dense family's traffic is the quality price; exp2 remains
      the prefill champion at 770-1000). Enable with
      DS4_QWEN4_EXPERIMENTAL_Q4_0_ROUTED=1 alone. The session's disk
      cleanup deleted nine obsolete experiment clones (~711 GB logical:
      the aborted q40routed-exp, q4dense-exp, q4-output-q4_0, q4qsa-exp,
      q4gdn-exp, and the four q2-q4* hybrids), keeping v3 (donor), exp2
      (speed profile), q2-v1 (Q2 artifact), and the imatrix.
      NOTE: /tmp cleanup during this session removed the
      OpenRouter API fixture and its TSVs (numbers preserved here;
      regenerate with the command in item 22i) and the 60-case
      qwen4-nll-long fixture; the BF16 fixture and its pack TSVs
      survive — archived durably at speed-bench/qwen38-bf16-reference/
      (fixture-100/, scores/ for all four pack TSVs, the two generator
      scripts, and the single-sequence verification set), since /tmp
      is volatile on this machine.
   k. VISION VERIFIED END-TO-END (first live check; previously only the
      Metal kernel fixtures covered it): ds4-server with the exp2
      vision sidecar (333 tensors, 27 blocks, Q8_0 projections; the
      only surviving pack that carries one) answered a synthetic
      shapes-image request correctly at temperature zero — "A red
      square, a green circle, and a yellow triangle are arranged on a
      dark background" — via OpenAI-style image_url data-URI content
      parts. Operational notes: the server defaults to thinking mode,
      which consumes max_tokens inside the reasoning block and returns
      an EMPTY content — pass "thinking": false for short multimodal
      answers; multimodal requests stay outside the disk prefix cache
      by design. GAP CLOSED: --rebuild-vision was run on the
      q40routed-v3dense-exp pack (reads the official visual shard
      model-00001, writes the 459.7 MB sidecar, merges tensor records
      atomically; pack_id unchanged) and the same live image test
      answers identically on the quality profile — both surviving
      Q4 profiles are now fully multimodal. The v3 pack itself still
      ships without the vision file (its manifest lists the artifact;
      loading tolerates the absence — rebuild the same way if needed). Regenerate the fixture with
      collect_official.py --model qwen/qwen3.8-flash --endpoint
      https://openrouter.ai/api/v1/chat/completions --api-key-env
      OPENROUTER_API_KEY --count 100 --max-tokens 24 --top-logprobs 5
      --token-limit-field max_tokens --thinking disabled. The exact-
      checkpoint BF16 anchor remains open: the transformers qwen4_exp
      checkout at /tmp/transformers-qwen4 (sparse checkout repaired;
      PYTHONPATH-shadowing needed because its wheel build drops
      transformers.utils) verifies DS4's segmented chat tokenization
      60/60 on /tmp/qwen4-nll-long, but the 354 GB MPS load exits
      silently at ~4 percent — debug device placement or CPU-layer
      streaming before trusting it.


23. SESSION SIXTEEN (this round) reopened prefill on the QUALITY PROFILE
   (q40routed-v3dense-exp), attributed its 8K chunk with the GPU stage
   profiler, falsified the round's stated premise, and landed two
   measured, fixture-gated wins — both default-on with kill switches.
   a. THE ATTRIBUTION FALSIFIED THE PREMISE: at 8K chunks the v3dense
      profile's GPU budget is 8.42 s busy — exp2-class, NOT 2x.  Per
      label: dense Q8_0 family 2.90 s (507 cb "model Q8_0 projection"
      + 32 cb BF16-activation variant), routed Q4_0 MoE 2.18 s, QSA
      attention 1.37 s, scalar BF16 family 0.77 s (120 cb = 48 router
      + 72 decay/beta), GDN R4 0.35 s, streaming top-k 0.25 s, HC
      family 0.32 s.  The Q8_0 dense family costs the SAME GPU time
      exp2 spends on its Q4_K dense: the tiled kernel is compute-bound
      at ~22 TFLOP/s effective (GDN qkv 8192x2560x12288 = 515 GFLOP in
      23.5 ms).  Dense PRECISION is not what separates the profiles'
      prefill at 8K chunks; re-quantizing dense was already off the
      table, and kernel headroom there is at the machine ceiling.
   b. THE REAL 421-510 GAP IS COLD PLE PAGES: the quoted number is a
      first-pass, cold-page-cache measurement.  A fresh text region's
      8K chunk stalls 6.9-7.7 s at the layer-1 PLE consumption points:
      131,072 rows x 100 B scattered over the 30 GB sidecar, read ONE
      ROW PER pread AT QUEUE DEPTH ONE by qwen4_ple_gather_pread_bf16
      (~59 us/row), with the 16 per-tile waits effectively serial
      (each next tile's gather starts only after the previous wait
      returns; only ~1 ms of upload/dispatch covers it).  The SAME
      chunk re-run warm is 987 tok/s (gather 204 ms).  The 4K-interval
      sweep reproduces the quoted band exactly on cold frontiers
      (451-488) while frontiers warmed by earlier runs measure
      921-991; exp2's 770-1000 quotes were warm from repeated runs.
   c. LANDED, DEFAULT ON — parallel PLE pread gather
      (DS4_QWEN4_PLE_GATHER_THREADS, default 32; 1 restores the serial
      loop; engages only at >= 512 rows so 16-row decode/MTP gathers
      are unchanged).  One tile's rows split across threads writing
      disjoint output slices — staged bytes identical by construction;
      full-model 16K/128 greedy dumps are SHA-identical between
      threads=1 and 32.  Cold thread sweep on fresh corpus regions:
      serial 557 -> 933/971/997 tok/s at 8/16/32 threads (gather
      6857 -> 922/533/338 ms); 64 showed no further gain; warm neutral
      (gather 200 ms, hidden under the 8.3 s GPU wall).  Paired cold
      A/B on adjacent fresh regions, back to back: 637 -> 1054 tok/s.
   d. LANDED, DEFAULT ON — tiled BF16 control-tensor matmuls
      (kernel_mul_mm_bf16_f32 in metal/dense.metal: the mul_mm
      template instantiated over 64-byte BF16 blocks dequantized to
      F16 tiles, dispatched from ds4_gpu_qwen4_bf16_matmul_model at
      >= DS4_QWEN4_BF16_MUL_MM_MIN_ROWS rows, default 32; 0 restores
      the scalar kernel; rows=1 decode stays scalar).  Serves the MoE
      router (48 calls, K=2560 N=512) and GDN decay/beta (72 calls,
      N=48).  Fixture test_model_bf16_matmul_rows pins the tiled path
      against a CPU reference that models the F16 operand rounding
      (bit-exact at fixture magnitudes — every product is exactly
      representable in F32, so both accumulation orders converge to
      the same sum) and the below-threshold scalar path against the
      exact reference, across aligned/bounds-checked out_dims and
      row counts.  Stage accounting: 766.6-800.9 ms -> 65-66 ms per
      8K chunk (0.55 ms/cb, 120 cb, distinct label "Qwen BF16 tensor
      matmul" so profiler kernel identity is unambiguous).
   e. MEASURED, FULL-MODEL (same binary, base via the two kill
      switches): warm 8K chunk 970/967 -> 1053/1047 tok/s
      (+8.2-8.3%); warm 4K-interval 8-frontier sweep, two interleaved
      pairs: 883.5 -> 948.7 and 878.5 -> 928.0 tok/s means (+7.4/+5.6
      percent); decode unchanged (32-35 tok/s; pair-2 columns
      identical to 0.1).  Post-change warm 8K budget: dense Q8_0
      2.95 s, MoE 2.27 s, attention 1.43 s (exhausted, item 21), GDN
      R4 0.35 s, top-k 0.26 s, HC 0.34 s, BF16 0.065 s; GPU busy
      7.96 s.  make test-qwen4 aggregate and test-qwen4-release-core
      green; make test-qwen4-release remains the pre-commit gate.
   f. QUALITY GATES: score_official vs the exact-checkpoint fixture —
      shipped defaults 0.044396 target MAE (archived baseline 0.0444;
      1/100 cases differs by 0.0004, the branch's known run-to-run
      boundary class; case-mean delta +0.000004).  The fixture's
      24-26-token prompts cannot gate a >=32-row path, so a second run
      with DS4_QWEN4_BF16_MUL_MM_MIN_ROWS=16 forced the tiled path
      onto the fixture prefills: 0.044025 (delta -0.0004, one
      first-token flip in 2040) — 10x inside the 0.002-0.005 budget.
      Full-scale calibration at a 16K frontier: tiled off vs on moves
      logits by 0.073 mean / 0.53 max (argmax unchanged there; the
      128-token greedy continuation diverges at token 15) — the SAME
      envelope as the branch's already-accepted chunk-size effect,
      re-measured same-build at 2048-vs-8192 chunks on this profile:
      0.074 mean / 0.55 max.  Both TSVs archived (r16,
      r16-tiledbf16).  The PLE change needs no drift gate (SHA-proven
      byte-identity).
   g. PROCESS LESSONS (each cost a false reading this session):
      (1) COLD-VS-WARM PLE IS A FIRST-CLASS BENCHMARK VARIABLE.  The
      exp2-vs-v3dense "prefill gap" quoted last round was mostly page
      cache state: exp2 was warm from repeated runs, v3dense's
      421-510 was the fresh clone's first pass.  Run sweeps twice and
      label pass 1 cold; cold test regions are cheap (byte-slice the
      corpus — disjoint token ranges hash to disjoint PLE rows).
      (2) zsh DOES NOT WORD-SPLIT unquoted variables: a shell variable
      holding "A=1 B=2" passed unquoted becomes ONE malformed argument
      and the kill switch never reaches the binary — two A/B "ties"
      this session were exactly that (the "base" run silently used
      defaults and ran at new-config speed).  Verify a kill switch by
      an observable effect (stage label, gather milliseconds), never
      by exit status alone.  Same family as 22f(1).
   h. REMAINING, IN MEASURED ORDER: the dense Q8_0 family is AT the
      ~22 TFLOP/s mul_mm ceiling — only a categorically different
      kernel moves it, and its GPU time is not this profile's quality
      price anyway; the QSA streaming top-k scan remains the
      long-context lever (item 22h, unchanged); MoE 2.27 s is at the
      post-fix ceiling.  Machine-state variance reached +-10 percent
      on absolute tok/s within minutes this evening — interleaved
      pairs or stage accounting, never single runs, are the evidence
      standard (re-confirmed item 22d's thermal-drift lesson).


24. SESSION SEVENTEEN (this round) attacked the last named prefill lever —
   the QSA streaming top-k candidate scan at long-context frontiers —
   with two bit-exact kernels now ON BY DEFAULT, and closed the dense
   Q8_0 "ceiling" question with a negative result.
   a. ATTRIBUTION (new ablation knob DS4_QWEN4_QSA_TOPK_ABLATE=score|
      merge, plus DS4_BENCH_QSA_VISIBLE_BLOCKS on qsa-prefill-bench to
      drive the last-chunk geometry — every query seeing 65536 pooled
      blocks): the scan cost 963 ms/layer per 512-query microtile set,
      SPLIT 734 scoring + 253 merging.  The scorer ran at 0.37 TFLOP/s —
      ~10x under the scalar ALU floor — because its grid re-read the
      2 KB query vectors per four-block threadgroup (q traffic ~8x the
      pooled-key traffic).  The merge full-sorted all 2048 threadgroup
      slots (66 barrier stages) per query per 1024-block tile to keep a
      sorted top-512 that a threshold filter would prove mostly empty.
   b. LANDED, DEFAULT ON — kernel_qwen4_qsa_score_tile_batch_bf16
      (DS4_QWEN4_QSA_SCORE_BATCHED=0 restores the original grid): one
      threadgroup loads the query vectors into registers once and scans
      eight blocks per simdgroup (32 per group), preserving each
      (query, block) dot's fma-chain order, simdgroup reduction, head
      max/sum, rsqrt, masking, and output placement BIT-IDENTICALLY.
      Guarded to BF16 pooled caches with 4 heads x 128 dims (F32/quality
      path untouched).  Scoring: 734 -> 138 ms/layer at 64K blocks.
   c. LANDED, DEFAULT ON — kernel_qwen4_qsa_merge_select_f32
      (DS4_QWEN4_QSA_MERGE_SELECT=0 restores the full-sort merge): drops
      every tile entry worse than the running 512th-best (provably
      outside the final top-512 — all 512 kept candidates beat it),
      compacts survivors via a threadgroup atomic, sorts only the
      survivor span (adaptive width, typically 64 slots), and merges
      with the sorted running set via the ten-stage bitonic merge after
      reversing the survivor block.  First tiles and >512-survivor tiles
      fall back to the in-kernel legacy full sort (rebuilt from global
      memory — the compaction may have copied survivors over their own
      originals, so stale tail entries must not reach the sort; that
      aliasing bug was caught on review before it ever ran).  Merging:
      253 -> ~60 ms/layer.  The comparator and total order are
      unchanged, so the ordered output is byte-identical — pinned by
      test_qsa_streaming_topk (batched scorer, both grids, tail guards)
      and the new test_qsa_streaming_topk_merge_select (production
      geometry keep=512/tile=1024, per-capacity baselines, CPU
      reference).
   d. MEASURED: standalone stage at 64K visible blocks 963 -> 372
      (scorer only) -> 177 ms/layer (both kernels, 5.4x; additive).
      Full model, v3dense pack: 4K-interval sweep (frontiers 4K..32K,
      two interleaved pairs) 947.6/930.9 -> 1004.3/994.7 tok/s means
      (+5.9-6.9 percent — the scan is context-proportional, so even
      short frontiers win); the whole-258048-token frontier 625.9 ->
      937.3/938.5 tok/s (+50 percent; last-chunk top-k stage ~11.4 s ->
      ~2.1 s, confirmed in-situ at 10.78 ms per microtile-scan vs the
      standalone 11.06).  Decode unchanged (rows=1 uses the separate
      m1 path; 30.0 vs 31.0 tok/s at 258K, within drift).  Byte
      identity: the 16K/128 greedy dump at DEFAULT settings is
      SHA-identical to the pre-change archive (f6c2a929...), so no
      drift gate is required — both kernels are exactness-preserving
      reorganizations.  make test-qwen4 aggregate green.
   e. NEGATIVE RESULT (probe, knob removed): routing the Qwen dense Q8_0
      prefill through the GLM nax_direct_rhs tile family
      (kernel_mul_mm_q8_0_f32_nax_direct_rhs_n128 at aligned shapes) is
      EXACTLY NEUTRAL — 2624.6 vs 2628.2 ms for the "model Q8_0
      projection" label on the same 8K chunk.  The standard mul_mm is
      genuinely at the ceiling for these tall (8192-row) batches; the
      NAX variants earn their keep on GLM shapes and M=1 decode, not
      here.  Confirms item 23h's ceiling reading; the dispatch knob was
      removed (measured dead, like item 22e's staging).
   f. PROCESS LESSONS: (1) the zsh word-splitting trap from 23g(2) bit
      AGAIN this session — a knobs-vs-baseline "A/B" whose env rode an
      unquoted $EXTRA silently measured off-vs-off (626 vs 614, pure
      drift, and it initially LOOKED like a 2 percent regression).  The
      repair discipline that caught it: re-run with explicit inline
      assignments AND an observable engagement check (the per-cb stage
      rate).  Rule: every A/B variable goes inline, never through a
      shell variable.  (2) The GPU stage profiler's per-dispatch flush
      perturbs DECODE badly (25.3 vs 30.0 tok/s at 258K — ~300 tiny
      flushes per token): its ~0 percent overhead claim is prefill-only;
      never read a decode column from a profiled run.  (3) Fixture
      lessons: a bit-exactness fixture must capture its baseline from
      the same input family (an F32-pool baseline made a correct BF16
      kernel look wrong by exactly the pool rounding), and
      truncation-invariance expectations only hold when every query's
      visible count fits the truncated capacity.
   g. POST-CHANGE STATE (v3dense, defaults): warm 8K-chunk prefill
      1053-1087 tok/s (fresh chunks) with the scan now 205 ms/96-cb
      scale at fresh-chunk sizes; long-context 258048 frontier 937-938
      tok/s whole-run (was 387 in item 22h's pre-session-15 state, 626
      before this session's kernels).  Remaining 8K-chunk budget at
      ~60K visible blocks: dense Q8_0 2.6 s, MoE 1.2-2.1 s (context-
      flat), attention 0.8-1.4 s (grows with budget), top-k 1.0-2.1 s
      (now linear-scans at 5.4x better efficiency; further gains need
      the matmul-shaped scorer — see h), GDN 0.17-0.35 s, BF16 0.07 s.
   h. REMAINING LEVER (for whoever returns): the scorer still runs
      scalar FMA at ~1.4-2 TFLOP/s effective after the restaging; the
      dot structure (4 heads x 128 dims, shared pooled key) is a
      (512-query x 512) x (512 x blocks) GEMM family that the tensor
      cores could run ~5x faster — but F16 operand rounding changes
      scores, so that path is drift-gated, not byte-exact (same class
      as the session-16 tiled BF16 matmuls, which passed at -0.0004).
      The merge is now ~60 ms/layer and near its floor.  TAKEN IN
      SESSION EIGHTEEN (item 25): the tensor-core scorer landed at
      2.8-3.2x on the stage with +0.000008 drift and is default-on.


25. SESSION EIGHTEEN (this round) took the remaining lever — the
   tensor-core QSA index scorer — landed it as DEFAULT ON with a kill
   switch, and passed every gate; the round's quality cost is the
   smallest ever measured for a numerics-changing path on this branch.
   a. RE-ATTRIBUTION FIRST (the standing rule): qsa-prefill-bench at
      65536 visible blocks measured the scan at 199.5 ms/layer full,
      split 153.9 scoring + 56.7 merging (DS4_QWEN4_QSA_TOPK_ABLATE),
      with the fresh-chunk causal ramp at only 16.1 ms/layer — the
      scorer is the context-proportional term and the lever, exactly
      as item 24h predicted.  The scorer's 550 GFLOP/layer at that
      geometry ran at ~3.6 TFLOP/s effective.
   b. LANDED, DEFAULT ON — kernel_qwen4_qsa_score_tile_mm_bf16
      (DS4_QWEN4_QSA_SCORE_MM=0 restores the scalar batched scorer;
      DS4_QWEN4_QSA_SCORE_MM_MIN_QUERIES, default 16, keeps decode
      (rows=1 m1 path, untouched) and MTP verifier rows (2-5 queries)
      on the scalar kernel).  The four index heads of a query fold
      into the ROWS of one GEMM A[queries*4, 128] @ B^T[128,
      tile_blocks] — B is the shared pooled-key row of each block, so
      the whole stage is a plain tall GEMM family.  F32 query vectors
      and BF16 pooled keys stage to F16 threadgroup tiles exactly
      once (the kernel_mul_mm operand-rounding policy, same class as
      the session-16 tiled BF16 matmuls); products accumulate F32 on
      the simdgroup matrix units.  One 128-thread threadgroup owns a
      64x64 output tile (16 queries x 64 blocks); A and B^T stage one
      64-wide K-chunk at a time into 16 KB of threadgroup memory and
      the reduced C tile overlays the dead staging region.  The
      epilogue stages the 16 accumulator fragments per simdgroup
      through that C tile (fragment element ownership is never
      indexed — item 20e(1)) and cooperatively reduces each query's
      four head rows with the production max/sum/rsqrt arithmetic and
      causal masking, so the merge-select kernel and everything
      downstream are unchanged.  Fixture test_qsa_streaming_topk_
      score_mm pins it against a CPU reference that MODELS the F16
      operand rounding (_Float16 casts), across a partial M-tile (70
      queries = 280 head rows), ragged merge tiles (2548 = 1024+1024+
      500 and a 1025 = 1024+1 capacity pass), masked-capacity
      invariance across different tail raggedness, the below-
      threshold guard (5 queries must stay scalar, byte-identical),
      and an OBSERVABLE-ENGAGEMENT check (env=1 must change at least
      10 percent of scores — a silently-undispatched path fails).
   c. OCCUPANCY WAS THE WHOLE KERNEL GAME, not tile arithmetic: the
      first 64x32-tile version with separately allocated A/B/C
      staging (32 KB threadgroup) ran the scorer at 80.9 ms/layer;
      chunked K staging plus the C-tile overlay (16 KB) took it to
      47.8 — a 1.7x kernel improvement from memory footprint alone,
      with IDENTICAL math.  The wider-tile staging-amortization
      reasoning (64x64 full-K staging, fewer restages) had predicted
      the opposite direction and measured 75.8.  A 12 KB BM=32 probe
      measured neutral (46.3-46.6 vs 47.8).  The sweet spot matches
      the dense mul_mm's own footprint (~16 KB).
   d. MEASURED, STANDALONE (8192 rows, 64K visible blocks): scorer
      134-154 -> 46-48 ms/layer (2.8-3.2x, ~11.5 TFLOP/s effective);
      full scan ~200 -> ~89 (scoring 47.8 + merging ~57, partially
      overlapped); fresh-chunk ramp 16.05 -> 13.6 ms/layer (the MMA
      path also wins short contexts — no regression anywhere).
   e. MEASURED, FULL MODEL (v3dense pack, 8K chunks, inline envs,
      interleaved pairs): 4K-interval sweep (frontiers 4K..32K, two
      pairs) gains +0.2 percent at 4K growing to +1.6 percent at 32K
      (context-proportional, both pairs agree per-frontier to 0.1);
      decode identical (34.3-36.1 tok/s both configs).  The whole-
      258048-token frontier, four passes OFF/ON/ON/OFF: 979.6 ->
      1025.1 and 948.0 -> 1010.4 tok/s (+4.6/+6.6 percent; pair means
      963.8 -> 1017.7, +5.6 percent) with decode unchanged at
      31.3-31.6 tok/s.  The absolute band drifted warm vs session
      seventeen's 937 anchor (machine state, item 23h) — the paired
      deltas are the evidence.
   f. QUALITY GATES (the drift gate, per the round's contract):
      score_official vs the exact-checkpoint fixture at SHIPPED
      DEFAULTS (the path engages on the fixture's 24-26-token prompts
      at the 16-query threshold — no forcing needed): target MAE
      0.044404 vs the archived 0.044396 baseline — delta +0.0000082,
      250x inside the 0.002-0.005 budget and 50x smaller than the
      tiled-BF16 precedent's -0.0004.  Every agreement metric is
      identical to the anchor (first-token 85, LCP 14.92, top-1
      1972/2040, recall 92.8, pair 93.1); per-case NLLs DO differ
      (case 48: 4.90113 -> 4.90323), which is the full-scale
      engagement proof.  The kill switch scores 0.0443957 and its TSV
      is byte-identical to the scalar baseline.  16K logits
      calibration: off-vs-on moves logits by 0.083 mean / 0.66 max
      (argmax unchanged) against the branch's accepted chunk-size
      envelope of 0.074/0.55 — same class, ~1.13x.  NOTE: the
      frontier-state comparator's exact QSA-cache check diverges at
      layer 7 by one BF16 ulp (layer 3 is exact) — expected for a
      numerics-changing path, since rounding-scale drift propagates
      through the residual stream into later layers' caches; that
      comparator is only a gate for exactness-preserving comparisons
      (chunk sizes), and the drift gate is the decisive one here.
      16K/128 greedy dump: the kill switch reproduces the standing
      anchor f6c2a929... exactly; the new default's dump hashes
      35ac916d9472f2db303f069ee31bdcca69021227e39a24a8b903b73e9dd2e8ea
      and diverges from the old continuation at token 14 of 128 —
      the same class as tiled-BF16's token-15 divergence.  TSV
      archived as scores/qwen38-bf16-local-v3dense-q40-r18-scoremm.
      make test-qwen4 aggregate green (126 PASS lines, 0 failures).
   g. PROCESS LESSONS (this round's bites):
      (1) OCCUPANCY BEATS STAGING ARITHMETIC for short-K GEMMs on
      this GPU.  Two kernel variants with identical math and tile
      shape differed 1.7x purely on threadgroup bytes; the staging-
      amortization model (fewer, wider tiles) pointed the wrong way.
      For K=128-class GEMMs, budget ~16 KB threadgroup memory first
      (chunk the K staging, overlay scratch on dead staging), then
      tune tiles.
      (2) A DRIFT-GATED FIXTURE NEEDS AN ENGAGEMENT CHECK built in:
      every tolerance-based check passes when the new path silently
      never dispatches, because the scalar outputs also sit inside
      the F16-reference tolerance.  The fixture now requires env=1
      to observably change outputs.  Third appearance of the item-
      22f(1)/23g(2) family, now pinned inside the fixture itself.
      (3) The frontier-state comparator's exact QSA-cache section is
      a CHUNK-SIZE tool: numerics-changing paths legitimately
      diverge there by ulps after the first consuming layer, so use
      the logits-delta calibration plus the fixture drift gate
      instead (this round measured both; the fixture MAE is the
      contract).
   h. REMAINING, IN MEASURED ORDER (long-context 8K-chunk scan, post-
      change): merge-select ~57 ms/layer (near its floor, item 24),
      the MMA scorer 46-48 (~2x under the dense mul_mm's 22 TFLOP/s —
      the short K=128 limits per-tile MMA amortization; a llama.cpp-
      style quadrant-packed fragment layout might close part of it),
      gathered attention ~143 (exhausted, item 21).  At the 258048
      frontier the top-k stage is now ~1.0-1.1 s/chunk of a ~20 s
      last chunk; the dense Q8_0 family, MoE, GDN, and BF16 controls
      remain at their measured ceilings (items 23-24).  Decode is
      untouched by construction (m1 path, threshold-guarded).
   i. SESSION-18 FOLLOW-UPS (same round, owner-directed, all
      uncommitted for the parallel optimization session to land):
      (1) the ANE experiment was REMOVED wholesale (see the repository-
      state section below).  (2) The Q4_0 routed profile no longer needs
      DS4_QWEN4_EXPERIMENTAL_Q4_0_ROUTED=1: the loader now auto-detects
      the substitution (a v3/Q4_K-profile pack whose routed tensors are
      literally Q4_0 has one meaning) — qwen4_validate_layer_directory
      probes the actual routed tensor type, qwen4_routed_qtypes_valid
      records the detection, and kernel selection follows the pack (it
      already branched on tensor->type in qwen4_graph_moe; the env now
      only serves as the =0 kill switch restoring strict v3-only
      rejection).  VERIFIED: no-env --inspect accepts the pack (q4_0
      144 tensors), the no-env server prints MoE-down=Q4_0-expert-split
      and generates; env=0 still rejects; exp2 commands keep working
      (its pack carries Q4_0 routed so detection engages; only the
      Q4_DENSE env remains required there); host and Metal suites
      green.  This also fixes the pre-existing footgun where env=1 on
      a plain v3 pack would have selected Q4_0 kernels against Q4_K
      data.  The v3dense pack was uploaded to
      ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4 on HF
      (private; 113 GB, manifest digests verified; base artifact
      renamed Q4KExperts -> Q40RoutedExperts after proving the loader
      binds via GGUF-internal ds4.pack.artifact metadata, not
      filenames; the repo was renamed from the internal
      q40routed-v3dense profile name to the release name, and the
      release no longer requires the env).  STALE-BINARY LESSON (item 20e(3) again): the first
      no-env verification failed because ./ds4 itself was not rebuilt
      after the loader change — the server/bench/test targets were;
      rebuild EVERY binary a change applies to, and confirm engagement
      through an observable (the type-mismatch diagnostic) rather than
      assuming.
   j. OWNER DECISION (pending, execute after the NEXT optimization
      round — i.e. after item 26's work is committed): rename the
      branch qwen3.8-flash-next-q4 -> qwen3.8-flash-next, because
      smaller quant profiles (Q2 and any future variants) will land on
      the same branch and the -q4 suffix is too narrow.  Deliberately
      NOT done yet — a parallel session is working on the branch and an
      early rename would break its upstream tracking.  At rename time:
      (1) git branch -m qwen3.8-flash-next-q4 qwen3.8-flash-next &&
      git push -u origin qwen3.8-flash-next && git push origin --delete
      qwen3.8-flash-next-q4 — GitHub does NOT redirect renamed branches
      for fresh clones, so existing checkouts must switch manually;
      (2) update the "Branch:" line in this handoff and the HF README's
      branch mention (ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4) — the
      only two live references; the qwen3.8-flash-next-q4-* ARTIFACT
      filenames (vision/MTP/manifest) are manifest-bound pack names and
      must NOT be renamed, and the historical session records below
      keep the old branch name as history.


26. SESSION NINETEEN (this round) took the stated primary candidate —
   re-running the opt-in GQA MMA attention A/B on the leaner v3dense
   chunk — and the round closed with a decisive, fully-measured NEGATIVE
   RESULT plus one real bug fix: the kernel had never been computing
   attention at all, and the corrected kernel, while faster, cannot hold
   the quality class at any speed that beats the scalar path.
   a. RE-ATTRIBUTION FIRST (v3dense, warm 8K chunk, prefill-only
      profiler): GPU busy 7975 ms with sparse QSA attention at
      1464.6 ms (12 cb, 122.0 ms/cb, 18.4 percent) — the third stage
      behind the dense Q8_0 family (~3.0 s) and MoE (2.35 s).  The
      session-14 absorption premise no longer holds: the walls that ate
      the isolated win (MoE tiling, tiled BF16, scan kernels) are gone.
   b. FULL-MODEL A/B ON THE AS-SHIPPED KERNEL (interleaved, inline envs):
      4K-32K sweep pair means 1012.4 -> 1046.3 tok/s (+3.35 percent,
      16/16 per-frontier comparisons positive); the whole-258048-token
      frontier 952.5 -> 1030.3 pair means (+8.2 percent, OFF/ON/ON/OFF),
      decode unchanged — a REAL end-to-end win, exactly as the leaner
      chunk predicted.  THE DRIFT GATE THEN FAILED CATASTROPHICALLY:
      fixture target MAE 0.1310 vs the 0.0444 anchor (worse than the
      all-Q4_K pack), first-token 85 -> 45, per-case NLLs 100/100
      changed, 16K logits 0.91 mean / 6.33 max, greedy diverging at
      token 0.  The control run reproduced the anchor exactly
      (0.044404), so the scorer invocation was sound.
   c. ROOT CAUSE (proto magnitude sweep plus staged-tile, score, and
      fragment dumps through unused output rows): the sfrag -> s_tile
      store loop was NOT guarded by is_score, so the four PV
      simdgroups' UNINITIALIZED sfrag fragments overwrote the score
      groups' freshly stored products every tile — their row_half maps
      them onto the same s_tile rows.  The kernel had computed EXACTLY
      UNIFORM softmax since session fourteen.  Every gate passed because
      every fixture validates at tiny magnitudes (fixture |q| <= 0.12,
      scores ~+-0.2) where uniform weights approximate true weights,
      and the fixture's own tolerance (5e-3 + 3e-2|ref|, later 2e-3
      absolute) swallowed the residual; the single-token boundary cases
      are additionally immune (scores cannot matter with one selected
      token).  The discriminating probe was a TWO-token row with real
      score spread: kernel scores read exactly 0.0 where truth was
      +5.5/-4.4.
   d. FIX AND HARDENING (all landed): the store loop is guarded in both
      the production kernel and the proto; the Metal fixture
      test_qsa_attention_gqa_mma gained a production-magnitude pass
      (x12 input scale, score spreads ~+-20) that negative-tests the
      bug (with the guard disabled the suite fails — note the old tiny
      case only FLAKED, because the clobber wrote nondeterministic
      garbage that happened to read as zeros on the passing runs);
      qsa_mma_proto gained a permanent magnitude sweep (exact,
      F16-modeled, and Q-split-modeled CPU references) plus the
      single/two-token boundary rows.
   e. THE FIXED KERNEL IS CORRECT AND STILL FASTER — AND STILL OUT OF
      QUALITY CLASS.  Correctness: fixture max error vs the scalar
      kernel 1.16e-05 at the original magnitudes (better than the old
      1.9e-04, which was the clobber's luck), two-token scores and
      weights exact.  Speed: standalone 131.2 -> 91.1 ms/layer (1.44x);
      full-model sweep pair 1033.2 -> 1077.8 (+4.3 percent); 258048
      pairs +4.3 percent pair means with the reversed pair at +9.0
      percent (one pass pair measured neutral — machine variance, the
      reversed pair re-established the win).  Quality: fixture target
      MAE 0.0581 (+0.0137 over the anchor, 3-7x the 0.002-0.005
      budget; first-token 85 -> 71, LCP 14.92 -> 11.78) and 16K logits
      1.21 mean / 11.06 max (~15x the accepted 0.074-0.083 / 0.55-0.66
      envelope), greedy diverging at token 2.
   f. THE RESIDUAL DRIFT IS NOT Q PRECISION.  An F16 hi/lo Q-split
      variant (kernel_qwen4_qsa_attention_gqa_mma_qsplit_f32, opt-in
      via DS4_QWEN4_QSA_GQA_MMA_QSPLIT=1 on top of GQA_MMA=1; a second
      16x256 F16 residual tile plus doubled score-group products)
      restores Q to ~22 mantissa bits — and leaves the drift
      UNCHANGED: kernel-level mean drift vs scalar 2.630e-4 vs the
      plain variant's 2.644e-4, full-model fixture MAE 0.0582 vs
      0.0581, engagement proven by differing NLLs.  The binding
      residual is the F16 P-tile rounding plus MMA accumulation order
      versus the scalar kernel's F32 serial semantics — inherent to
      running softmax weights through F16 matrix operands.  Q-split
      also costs the entire win standalone (158.3 vs the scalar 131.2
      ms/layer), and a further P-split would cost as much again.
      CONCLUSION: F16-operand gathered attention cannot meet the
      quality envelope at a speed that beats the scalar kernel on this
      model; the path stays OPT-IN for study and the direction is
      CLOSED (the scalar kernel's item-21 "exhausted" verdict now
      rests on honest numbers).
   g. STATE AT HANDOFF: default path byte-identical throughout — the
      16K/128 greedy dump at defaults hashes 35ac916d... and
      DS4_QWEN4_QSA_SCORE_MM=0 reproduces f6c2a929... on the final
      binary; make test-qwen4 aggregate green; git diff --check clean.
      New/changed for this item: metal/qwen4.metal (guard + qsplit
      instantiation via a QSPLIT template parameter), ds4_metal.m
      (QSPLIT env + scratch sizing + pipeline selection),
      tests/test_qwen4_metal.c (production-magnitude and Q-split
      fixture passes with a mean-drift family check),
      speed-bench/gqa_mma_proto.{m,metal} (magnitude sweep, Q-split
      variants, store-guard fix; debug dumps behind args.debug).
   h. PROCESS LESSONS (each cost real time or a false reading):
      (1) THE ITEM-21C MAGNITUDE LESSON, SHARPER FORM: any
      softmax-bearing kernel must be validated at score spreads where
      the softmax is NOT uniform — a uniform-softmax bug is invisible
      both at small magnitudes AND at single-token selections; every
      attention fixture needs a two-token-plus, large-spread case
      (the hardened fixture now carries one).
      (2) Uninitialized-fragment clobber is NONDETERMINISTIC garbage:
      the same fixture passed at 1.9e-04 and failed at 2.3e-03 across
      runs — flaky tolerances on an unchanged kernel are a bug signal,
      not noise.
      (3) REBUILD EVERY BINARY THAT LINKS THE METALLIB, quality tools
      included: one Q-split "identical result" this session was a
      stale score_official binary (its own make target,
      make -C gguf-tools quality-score) silently re-measuring the old
      config — the fourth stale-binary incident, extending item
      20e(3)/25i's family from benches to scorers.
      (4) Dump-based bisection through unused output buffer rows beat
      three rounds of code inspection: the invariant pair "single-token
      row perfect, two-token row wrong" plus staging/fragment dumps
      localized the clobber in minutes; keep that probe pattern in the
      proto.
      (5) NEAR-TIE ARGMAX FLIPS make per-element tolerances meaningless
      at spiky fixtures (a single ulp decides the dominant token): gate
      such paths on MEAN error plus a structural small-magnitude pass,
      never on per-element maxima alone.
      (6) SINGLE-PAIR 258K SWINGS: one OFF/ON pair measured neutral
      the same evening a reversed pair measured +9.0 percent — the
      interleaved-pairs rule (item 23h) applies at every frontier,
      including (especially) the long ones.

27. SESSION TWENTY (this round) answered the round's question on the
   QUALITY PROFILE (v3dense): replay-free MTP is a WORKLOAD-DEPENDENT win
   that the shipped scheduler already arbitrates correctly at the shipped
   margin.  The depth and margin optima were measured on this pack for the
   first time, the acceptance record was re-attributed from a pack property
   to a prompt-family property (with same-slice v3-pack controls), and NO
   SOURCE CHANGE was warranted: the working tree is byte-identical to the
   session-nineteen handoff (every number below is CLI/env configuration
   only; make test-qwen4_metal and the full Metal fixture suite,
   including the MTP capture/replay parity fixtures, re-ran green at the
   round's start on the rebuilt binaries).
   a. RE-BASELINE (short context, five interleaved ds4-server instances —
      plain plus depths 2-5, default scheduler, margin 110; requests of
      256 greedy tokens with ignore_eos over fresh ~148-token corpus
      slices; interleaving made possible by per-instance DS4_LOCK_FILE
      overrides on this 512 GiB machine, with engagement proven per
      instance by the "MoE-down=Q4_0-expert-split" graph line and the
      sidecar/capture-depth lines): plain decode 35.27-35.53 tok/s
      across blocks (band 34.6-36.0, consistent with the record's 34-36).
      MTP cycle components on this pack at short context: base target
      pass ~26-27 ms, MTP history 3.3-4.1 ms, draft ~2.6 ms per token
      (2.9/5.3/7.7/10.2/12.2 at depths 2-6), verify 34.5/41.0/48.4/54.5/
      60.8 ms (depths 2-6), commit 3.3-4.2 ms, restore-select <0.5 ms.
      The verify pass is 37-41 percent CHEAPER than the v3 pack's
      (48.4 vs 77.6 ms short / 82.4-82.5 ms at 8K, same depth 4): the
      Q4_0-routed substitution speeds the verify MoE exactly as it
      speeds decode.
   b. THE DEFAULT SCHEDULER BYPASSED EVERY corpus request, CORRECTLY.
      With the default scheduler at margin 110, all twelve short-context
      corpus requests (depths 2-5, three interleaved rounds) switched to
      target-only after the 16-cycle window at cumulative actual/baseline
      ratios 1.148-1.897 (several depth-4/5 runs tripped the 1.5x
      severe-loss check after 1-5 cycles); net 34.80-35.16 tok/s vs
      plain 35.53 — the bounded cost of the evaluation window.  Forced
      engagement (DS4_QWEN4_MTP_SCHEDULER=0) measured the true
      speculative cost on corpus prose: 32.6/31.3/29.1/28.3 tok/s at
      depths 2-5 — BELOW plain at every depth (accepted 0.59-0.84 of
      drafts).  On deterministic continuations the same forced runs WIN
      monotonically in depth: 39.6/46.0/49.9/54.6 tok/s at depths 2-5
      on the "one, two, three," continuation (accepted 1.88-4.54), and
      with the DEFAULT scheduler engaged (zero bypass events, engagement
      ratio ~0.75) the end-to-end numbers are d4 50.2, d5 54.2 tok/s —
      +42/+54 percent over plain, the fastest decode ever recorded on
      this machine for this model.  A factual-list prompt reproduces the
      win (d4 44.7, d5 48.1, d6 48.4; +26 to +37 percent, engaged), and
      the acceptance harness's own code-continuation prompt family is
      bypassed like prose (accepted 0.33-0.5).  DEPTH OPTIMUM ON THIS
      PACK (completed by the 5-8 extension in sub-item i): 7 on the most
      predictable family, 5-6 on mid-acceptance families; deeper than
      the exp2/v3 optima (3/4), consistent with the 37 percent cheaper
      verify amortizing deeper drafts.
   c. ACCEPTANCE IS A PROMPT-FAMILY PROPERTY, NOT A PACK PROPERTY — the
      round's decisive re-attribution.  The same ~148-token corpus slice
      on the ORIGINAL v3 pack (forced depth 4) regresses identically
      (22.92 vs plain 28.28 tok/s, accepted 1.18) while the chat
      continuation on v3 wins identically (35.90 vs 28.34, accepted
      3.41): pack-for-pack, v3dense and v3 behave the SAME on the same
      text.  The record's historic acceptance numbers (3.65 committed
      on v3, 2.65 accepted on the harness's mtp-10k+500 case, exp2
      engaged at 8.7K) are therefore properties of those prompt
      constructions — chat-continuation style, the harness's rotated
      corpus — not of the packs.  Concretely, the Qwen MTP draft matches
      the target argmax at roughly 50-60 percent per position on
      novel-prose continuation (any context length), near-perfectly on
      closed-form sequences, with code in between (0.33-0.5 on the
      harness's own code prompt).  MTP profitability follows acceptance
      mechanically; nothing about the v3dense pack degrades it.
   d. LONG CONTEXT (~7.7-8.5K-token corpus prompts, default scheduler,
      depths 2-5 interleaved with plain, ctx 16384): plain 33.06 tok/s;
      MTP 32.94/31.92/32.32/32.10 at depths 2-5 — every request bypassed
      (accepted 0.68-0.85), i.e. 0.4-3.5 percent below plain, again the
      bounded window cost.  The v3-pack control on the SAME slices:
      plain 25.90, default-scheduler d4 25.39 (bypassed), forced d4
      20.68 (accepted 0.90-0.94, verify 82.5 ms).  Two consequences:
      deep Italian-novel context does NOT rescue draft agreement on
      either pack (the 8.7K/10K engagement precedents were their slice
      constructions), and v3dense plain decode at 8K beats v3 by +28
      percent (33.1 vs 25.9) — the routed win carries at long context,
      measured here for the first time.
   e. MARGIN 110 VALIDATED FROM BOTH SIDES; the formal margin sweep is
      moot.  Profitable families sit at engagement ratio ~0.75 (any
      margin in [100,120] keeps them engaged); unprofitable ones bypass
      at 1.15-1.9 (no margin below ~200 keeps them engaged, and doing so
      would FORCE the measured 8-21 percent losses — item 8's lesson in
      reverse).  The margin's known bias (baseline prices one plain
      token at base-eval wall time, understating true per-token cost by
      ~5 percent here: measured base pass ~26.5 ms vs 28.1 ms wall per
      token) is exactly what 110 covers.  No change.
   f. SECONDARY RE-ATTRIBUTION of the ~300-kernel decode window on the
      current binary (relative-only, per the standing rule; the per-stage
      flush invalidates absolute throughput).  CPU submission buckets
      (DS4_QWEN4_PROFILE=1, plain decode, rows=1): dense 25.0-25.5 ms of
      a 26.3-26.9 ms total, recurrence 0.34-0.38, QSA 0.24-0.25, routed
      0.51-0.55, PLE/embedding/cache <0.2 combined — the dense family is
      ~95 percent of submission time, unchanged from the record.
      Synchronized stage profile (DS4_METAL_DECODE_STAGE_PROFILE=1 plus
      DS4_QWEN4_TINY_STAGE_PROFILE=1 on an engaged depth-4 server,
      layers 1-47): tokens=1 sums to 2.77 ms/layer split QSA 0.588
      (21 percent), MoE 0.524 (19), GDN 0.510 (18), MLP-HC-read 0.426
      (15), attention-HC-read 0.390 (14), MLP-HC-write 0.300 (11),
      attention-HC-write 0.033 (1); tokens=4 (verifier rows) sums to
      3.35 ms/layer — every stage scales sub-linearly (MoE 1.40x, GDN
      1.36x, QSA 1.21x, HC reads 1.06-1.07x for 4x rows).  This
      CONFIRMS the items 8-12 conclusion set on the current binary: the
      M=1 window is spread across seven comparable latency-bound stages
      with no dominant target, and the verify pass is the same structure
      at modestly worse parallelism — no new kernel direction is
      justified, and none is proposed.
   g. LANDED CONFIGURATION (documentation-level; no code change): MTP on
      this profile is a scheduler-arbitrated workload bet — enable the
      sidecar with --mtp-draft 5 as the balanced default (within one to
      two percent of the best on every profitable family and cheapest on
      capture memory), 7 for workloads known to be highly predictable
      (counting/pattern/structured continuations; see sub-item i), and
      never 8 (measured worse on both engaged families); leave the
      scheduler and margin at defaults.  Upside on predictable
      continuations is +26 to +57 percent (56.3 mean / 57.8 peak at
      depth 7); worst-case cost on entropic prose is bounded by the
      16-cycle window at 0.4-3.8 percent.  Enabling MTP unconditionally
      as a blind default is NOT recommended for pure-prose completion
      workloads, where it never engaged on any measured slice.
   h. DEPTH EXTENSION 5-8 (owner-directed follow-up after the primary
      2-5 sweep; default scheduler, three interleaved rounds plus a
      two-round d8 probe, same prompt families): the depth optimum is
      FAMILY-DEPENDENT and the boundary is now measured.  Chat
      (deterministic continuation, plain 35.87): d5 54.25, d6 54.51,
      d7 56.27 (57.83 best run — the machine record for this model),
      d8 55.11 — monotone through 7, d8 below d7 in both rounds;
      acceptance 5.737/7 at d7 (82 percent per position), 5.919/8 at d8,
      i.e. the eighth row's ~6.6 ms verify plus ~2.6 ms draft has turned
      EV-negative on the BEST family.  Factual lists (plain 35.59):
      d5 48.27, d6 48.52, d7 47.82 — saturates at 5-6; at d8 the deeper
      cycles pushed the cumulative ratio past break-even and the
      scheduler BYPASSED every d8 factual request (34.73, ~plain) — the
      cost controller correctly refusing the deeper bet.  Corpus
      (bypassed at every depth, 34.55-35.00 vs plain 35.92): the window
      cost is NOT monotone in depth — the 1.5x severe-loss check trips
      after 4 cycles at d7 versus 10 at d6 and 16 at d5, so deeper
      drafts can cost LESS when bypassed.  The v3-pack depth-8
      rejection therefore TRANSFERS to v3dense, but for the opposite
      reason: there acceptance collapsed; here acceptance holds and the
      marginal verify row is simply not worth its latency past 7.
   i. PROCESS LESSONS (each generalizes):
      (1) PER-REQUEST MTP COUNTERS ARE CUMULATIVE PER GRAPH and the
      accepted-length histogram prints only at block commits 1 and
      multiples of 32 — short or quickly-bypassed requests never print
      one; use the per-cycle timing lines for acceptance on short runs.
      (2) PREFIX-CACHE CALIBRATION TRAP: probe prompts that are
      truncations of each other prefill a one-token suffix, which a
      suffix-parsing calibration reads as "1 token" before scaling the
      next probe absurdly (HTTP 400).  Calibration slices must be
      disjoint and the parse must read the prompt TOTAL.  Same family as
      22f(1)/23g(2): the first calibration iteration of this round was
      exactly this bug.
      (3) CONCURRENT ds4-servers need distinct DS4_LOCK_FILE overrides
      (the /tmp/ds4.lock flock singleton refuses a second instance);
      five interleaved instances with shared file-backed pack pages ran
      stable at 97 percent free memory and made tight same-minutes A/Bs
      cheap.  Engagement still verified by observables per instance.
      (4) The acceptance harness's "enable_mtp" request field is NOT
      parsed by the server (unknown JSON keys are skipped); MTP is
      engaged by loading the sidecar.  The field is vestigial.
      (5) Driver archived as speed-bench/qwen4_mtp_sweep.py (server
      lifecycle, interleaving, timing/histogram/bypass parsers,
      engagement assertions); round logs and JSONs in /tmp/ds4_mtp_round
      (volatile).

28. SESSION TWENTY-ONE (owner decision): the Q4_0-routed profile is the
   STANDARD implementation; the experimental framing and its environment
   variable are REMOVED, and the branch is committed and pushed to the
   fork.  This supersedes the old "keep experimental profiles guarded"
   instruction for the ROUTED side only.
   a. CODE (ds4.c): `DS4_QWEN4_EXPERIMENTAL_Q4_0_ROUTED` is gone —
      neither `=1` nor `=0` has any effect.  The v3/Q4_K profile now
      unconditionally accepts the all-Q4_0 routed variant
      (gate/up/down all Q4_0 — a mixed or partial substitution is still
      rejected exactly as before, and the v4/Q2 profile contract is
      untouched).  Detection from the pack's actual tensor types is
      unchanged (the manifest still declares profile=q4_k; the
      substitution is provenance-only metadata the loader ignores);
      `qwen4_experimental_q4_0_routed_disabled/enabled` became
      `qwen4_q4_0_routed_pack_active`, and the graph log keeps the
      `MoE-down=Q4_0-expert-split` observable.  The MTP SIDECAR contract
      is deliberately unchanged (strict Q4_K routed experts — a Q4_0
      MTP sidecar remains rejected; building one is still "recommended
      next work" #4, never measured).  `DS4_QWEN4_EXPERIMENTAL_Q4_DENSE`
      (the exp2 Q4_K-dense GDN/QSA experiment) is untouched and stays
      opt-in.  The repacker's provenance manifest key was renamed to
      `source_derived_q4_0_routed` for FUTURE repacks; existing packs
      (local and the HF upload) carry the old
      `experimental_source_derived_q4_0_routed` key, which nothing
      reads — both load identically.
   b. DOCS: QWEN38_FLASH_NEXT.md now describes Q4_0-routed as a fully
      standard, unconditionally accepted variant of the v3 pack format
      (the "v3dense" working name is retired from the technical doc;
      pack DIRECTORY names on disk are historical artifacts and keep
      their literal paths).  This handoff keeps its session history
      verbatim — the name appears below only as the historical record
      of which artifact was measured.
   c. VERIFICATION before commit: rebuild, the full Metal fixture suite,
      `make test-qwen4-release` aggregate, and live loads of all three
      surviving pack flavors on the final binary — the standard pack
      with no env (Q4_0-expert-split engaged), the same pack with the
      removed env set to 0 (must STILL load: the kill switch is gone,
      proving the removal), the original v3 pack (Q4_K routed, loads),
      and exp2 via `DS4_QWEN4_EXPERIMENTAL_Q4_DENSE=1` (loads).
   d. COMMIT/PUSH: one commit on `qwen3.8-flash-next-q4` pushed to
      origin (ivanfioravanti/ds4-metal), containing the session 12-21
      work plus this standardization; untracked tooling and the quality
      fixtures (speed-bench/qwen38-bf16-reference/, the repackers, the
      proto benches' SOURCES, the MTP sweep driver, and this handoff)
      are added; built bench BINARIES stay out via .gitignore.
      (The branch was later renamed to `qwen3.8-flash-next`.)
   e. MANIFEST RECIPE FIX (follow-up, owner-directed): the standard
      pack's manifest still carried the CONVERTER's recipe text
      `quantization.routed_experts: {qtype: Q4_K}` — stale prose from
      the original v3 conversion that the repacker had deliberately
      left; with the profile standardized this invited exactly the
      "are we sure this is Q4_0?" question (the FILENAME
      Q4KExperts-...gguf and the profile-name-driven startup line
      "Q4_K/Q4_K routed" read the same way).  BYTE-LEVEL CONFIRMATION
      (independent GGUF tensor-directory parse, on BOTH the Studio and
      the MacBookIvan Thunderbolt copy): the base file declares exactly
      144 tensors of ggml type 2 (Q4_0) = 48 layers x 3 routed
      gate/up/down, 315 Q8_0, 752 BF16, ZERO Q4_K; the only Q4_K
      tensors in the pack are the MTP sidecar's three (strict by
      design); the loader never parses the recipe block (verified by
      grep and by ds4 --inspect accepting the edited manifest with
      tensor_manifest_sha256 unchanged).  The manifest now records
      routed_experts {qtype: Q4_0, block_size: 32, note: ...} plus the
      `source_derived_q4_0_routed` provenance key; the same bytes were
      synced to the MacBookIvan copy (sha-verified) and UPLOADED to the
      HF repo ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4 (commit b617e1e4;
      the pre-upload remote manifest was verified byte-identical to the
      pre-fix local one first, and the post-upload download re-verified
      sha 61334043...).  The repacker now rewrites the recipe block for
      future repacks so this cannot regress.  Two Q4_K-looking artifacts
      deliberately remain: the base file's historical NAME (the loader
      binds via GGUF-internal ds4.pack.artifact metadata) and the
      startup bind line (it prints the pack PROFILE's declared routed
      names, not the bound tensor types — a known logging wart, fix
      offered but not requested).


## Complete source checkpoint

The full official BF16 model is downloaded in the Hugging Face cache. All 131
safetensor shard links resolve to regular files:

```text
/Users/ifioravanti/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540
```

`du` without dereferencing reports zero for this snapshot because its files
are symlinks into the Hub blob store. Use `find -L` or dereference the links
when validating the cache.

## Model artifacts

Original Q4 v3 pack:

```text
/Users/ifioravanti/models/qwen3.8-flash-next-q4-v3/Qwen3.8-Flash-Next-Q4KExperts-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out.gguf
/Users/ifioravanti/models/qwen3.8-flash-next-q4-v3/Qwen3.8-Flash-Next-PLE-Q4_1.gguf
/Users/ifioravanti/models/qwen3.8-flash-next-q4-v3/qwen3.8-flash-next-q4-mtp.gguf
```

Q2 experimental pack:

```text
/Users/ifioravanti/models/qwen3.8-flash-next-q2-v1/Qwen3.8-Flash-Next-IQ2XXSGateUp-Q2KDown-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out.gguf
/Users/ifioravanti/models/qwen3.8-flash-next-q2-v1/Qwen3.8-Flash-Next-Q2-PLE-Q4_1.gguf
```

Full Q4 dense experimental clone:

```text
/Users/ifioravanti/models/qwen3.8-flash-next-q4dense-exp
```

Current best experimental clone, with source-derived Q4_K dense GDN/QSA and
source-derived Q4_0 routed base experts:

```text
/Users/ifioravanti/models/qwen3.8-flash-next-q4dense-q40routed-exp2
```

It contains the base, PLE, vision, MTP, and manifest files. Enable it with:

```text
DS4_QWEN4_EXPERIMENTAL_Q4_DENSE=1
DS4_QWEN4_EXPERIMENTAL_Q4_0_ROUTED=1
```

There is an older partial/aborted clone at:

```text
/Users/ifioravanti/models/qwen3.8-flash-next-q4dense-q40routed-exp
```

Do not delete it without first explaining the deletion and obtaining the
required authority.

## Accepted implementation work

The current working tree includes these accepted changes:

1. Replay-free Qwen partial MTP commits. The verifier captures selectable
   intermediate GDN and PLE states, and partial acceptance selects the target
   state directly instead of replaying accepted target rows. Timing logs show
   `path=direct-partial` and `replay=0`.
2. BF16 replay-free recurrent-state capture, including exact capture/restore
   tests.
3. Runtime-derived Q8_0 hyper-connection matrices. This was a major Q2 speed
   improvement and remains enabled for the Q4 profile.
4. Eight layers per Metal command buffer, with an early layer-zero flush for
   SSD PLE overlap.
5. BF16 GDN recurrent state for ordinary and MTP decode.
6. Partial and full GDN decode fusion, HC-down SiLU fusion, HC-up/mix fusion,
   and Q8 output-plus-HC-write fusion for ordinary decode.
7. Source-derived Q4_K dense GDN/QSA support under
   `DS4_QWEN4_EXPERIMENTAL_Q4_DENSE=1`.
8. Source-derived Q4_0 routed base-expert kernels, loader support, converter,
   and exact CPU-reference tests under
   `DS4_QWEN4_EXPERIMENTAL_Q4_0_ROUTED=1`.
9. Q2 conversion and the expert-down rows-2 kernel.
10. `/v1/completions` `ignore_eos` parsing was fixed.
11. The Qwen quality scorer now accepts `--ple FILE`, and its Makefile target
    links `ds4_qwen4.o` on Darwin and Linux.
12. The Qwen MTP request scheduler now uses a default observation window of
    16 cycles instead of 4, avoiding premature disablement on a cold prefix.

## Provisional tiny-verifier fusion work at handoff

The most recent change generalizes four exact row-1 fusions to tiny Qwen
batches up to `DS4_QWEN4_MTP_MAX_DRAFTS`:

- Q8_0 projection plus SiLU
- paired Q8_0 gate/up projection plus SwiGLU
- Q8_0 hyper-connection up projection plus sigmoid stream mix
- Q8_0 output projection plus hyper-connection stream write

Files involved:

```text
ds4.c
ds4_gpu.h
ds4_metal.m
metal/qwen4.metal
tests/test_qwen4_metal.c
```

The original row-1 kernels are still selected for ordinary decode. Tiny-batch
fusions can be disabled for an A/B run with:

```text
DS4_QWEN4_TINY_BATCH_FUSIONS=0
```

Hot-cache A/B results on the same binary and prompt:

```text
tiny fusions off: verifier4 48.84, 49.03 ms; decode 41.34 tok/s
tiny fusions on:  verifier4 47.71, 47.89 ms; decode 41.78 tok/s
```

This is a small but repeatable improvement: about 1.1 ms or 2.3% from a
depth-4 verifier. The generated text was identical in the A/B.

The full Metal fixture suite passed after the new kernels were added, but the
existing fixture calls exercise their row-1 entry points. Before treating this
work as fully accepted, extend `tests/test_qwen4_metal.c` with explicit 3- or
4-row parity tests for all four new generic kernels. Compare the fused outputs
with the existing standalone kernels or a CPU reference. Then rerun the full
suite.

A diagnostic-only environment variable was also added so the existing layer
stage profiler can inspect verifier rows:

```text
DS4_QWEN4_TINY_STAGE_PROFILE=1
```

It has no effect unless a Metal decode-stage profile is requested.

## Performance results

### Ordinary decode

- Original Q4/Q8-dense model: about 28 tok/s.
- Full Q4_K dense GDN/QSA clone: about 29 tok/s, roughly +3.9%.
- Q2 pack: about 34 tok/s.
- Q4_K dense plus Q4_0 routed experts: about 34.7–35.0 tok/s.
- The Q4_0 routed-expert change is responsible for the largest retained Q4
  gain, about +23% over the Q4_K routed control.

The Q4_0 routed microbenchmark measured:

```text
Q4_K: 0.2188 ms/layer
Q4_0: 0.0692 ms/layer
speedup: 3.16x
```

### Exact replay-free MTP

Depth 4 remains the best tested fixed depth:

```text
depth 2 median decode-only: 36.27 tok/s
depth 4 median decode-only: 38.45 tok/s
depth 8: 28.65–36.95 tok/s, rejected
```

With the latest tiny-verifier fusions and a favorable short prompt, decode
reached 41.78 tok/s. This is not a representative 60 tok/s result.

Typical hot depth-4 full-accept cycle after the latest fusion work:

```text
initial target decode: ~28.7–29.2 ms
snapshot:             ~0.5–0.8 ms
draft chain:          ~7.4–7.8 ms
target verifier4:     ~47.7–47.9 ms
MTP history/commit:   ~4.7–5.1 ms
full cycle:           ~89.7–90.0 ms for 5 target tokens
```

A 60 tok/s full-accept cycle must be no more than 83.3 ms, so the current hot
full-accept path still needs roughly 6.5–7 ms removed. Partial acceptance and
first-draft misses reduce average throughput further.

The depth-4 scheduler with its new 16-cycle window behaved well on mixed
prompts: it retained MTP on profitable prompts and eventually bypassed it on a
prompt with a cumulative loss. Keep exact target validation; the experimental
trusted-draft mode produced degenerate output and was rejected.

## Quality result for Q4_0 routed experts

The repository scorer was run on a fixed 100-case continuation fixture using
the same Qwen tokenizer and target strings for both profiles:

```text
Q4_K routed:
  target tokens: 2,228
  average NLL: 0.754021252
  first-token match: 34
  average LCP: 1.930

Q4_0 routed:
  average NLL: 0.760319680
  first-token match: 37
  average LCP: 2.320

NLL delta: +0.006298428 (+0.835%)
case wins: exactly 50/50
```

Interpret this as a small real quality tradeoff suitable for an opt-in Q4
performance profile, not an automatic default. Ephemeral detailed results may
still be present at:

```text
/tmp/qwen4-q4k-nll-100.tsv
/tmp/qwen4-q40-nll-100.tsv
```

The built-in `ds4-eval --nothink --tokens 256` was not a valid quality gate for
this model because verbose generations were truncated before final answers.

## Profiling conclusions

- The GPU is saturated during decode; model-weight streaming dominates.
- PLE is SSD-backed and CPU-staged, but its read is overlapped with GPU work.
  PLE and CPU n-gram lookup are not the ordinary decode bottleneck.
- Current hot ordinary rows show effectively zero PLE layer-1 wait.
- The Q4_0 routed expert stage is now a small fraction of a token.
- In a depth-4 verifier, the aggregate `dense` bucket is about 46–47 ms of the
  47–48 ms total verifier. This is the remaining dominant target.

Synchronized stage profiling of layer 24 showed the scaling problem clearly:

```text
stage                 M=1       M=4
attention HC read     ~0.36 ms  ~0.52 ms
GDN                   ~0.46 ms  ~1.61 ms
attention HC write    ~0.26 ms  ~0.40 ms
MLP HC read           ~0.47 ms  ~0.49 ms
MoE                   ~0.50 ms  ~0.70 ms
MLP HC write          ~0.26 ms  ~0.34 ms
```

These synchronized numbers perturb scheduling and must only be used
relatively. The GDN tiny-batch path is the clearest per-layer scaling target.

Forcing the existing Q8_0 rows-8 weight-reuse kernel at MTP sizes with
`DS4_QWEN4_Q8_0_EXACT_MIN_ROWS=2` was flat to slightly slower. Do not make that
the default based on the current evidence.

## Rejected experiments

Do not repeat these without a new hypothesis:

- trusted-all MTP: faster but degenerate quality
- Q4_0 output head
- full PLE warmup
- direct GPU PLE gather
- shared/routed expert overlap
- larger GDN BV tiles, including BV=32
- Q4_K GDN concatenation
- Q4 attention-output plus hyper-connection fusion
- Q4_0 hyper-connections
- Q4_0 dense GDN/QSA; dedicated kernels were 29–33% slower than Q4_K
- forcing the Q8_0 rows-8 kernel for tiny verifier batches
- MTP depths 2 and 8 as fixed defaults

The abandoned dense-Q4_0 kernel/API experiment was fully removed. The symbols
`qwen4_q4_0_matmul`, `kernel_qwen4_q4_0_f32`,
`qwen4_q4_0_row_bytes`, and `dispatch_q4_0` should remain absent. Q4_0 routed
expert symbols are expected and retained.

## Recommended next work

### 1. Finish the tiny-verifier fusion gate

Add explicit multirow correctness fixtures, run the full Metal and release
tests, and retain the change only if the measured improvement remains. This is
the smallest unfinished obligation in the current tree.

### 2. Batch MTP history reconciliation into one MTP pass

`qwen4_graph_mtp_history_after_target()` currently executes the first accepted
row as an M=1 pass using `mtp_trunk_last`, then executes the remaining accepted
rows as a second batched pass using prior target trunk rows. On a full depth-4
accept this costs around 5 ms.

Build a temporary widened input block containing:

```text
[saved mtp_trunk_last, target_trunk_row0, target_trunk_row1, ...]
```

Then call `qwen4_graph_mtp_run_with_ids()` once for all accepted rows. Preserve
the final authoritative target trunk row for the next cycle. A likely scratch
candidate is `g->block`: command ordering can let the initial hidden norm read
the assembled input before later MTP stages reuse that buffer. Verify aliasing
carefully; adding a dedicated small scratch tensor is safer if necessary.

This is attractive because saving 1–2 ms would apply to every profitable MTP
cycle without changing acceptance quality.

### 3. Implement the dedicated inner-64 GDN tiny-batch algorithm

The synchronized layer profile shows M=4 GDN scaling much worse than the
surrounding stages. Profile the existing Q4_K projections and the capture-aware
GDN recurrence separately, then implement a dedicated tiny-batch/inner-64 path
only if a kernel-level A/B identifies the source. Keep row-1 unchanged.

### 4. Try a source-derived Q4_0 MTP routed sidecar

The full BF16 source is now local. Repack only these three MTP expert tensors
to standard Q4_0:

```text
language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight
language_model.mtp.layers.0.mlp.switch_mlp.up_proj.weight
language_model.mtp.layers.0.mlp.switch_mlp.down_proj.weight
```

The current MTP binder is strict Q4_K and must be guarded explicitly before
loading such a sidecar. The existing Q4_0 routed Metal kernels can execute the
MTP MoE once binding accepts it. Expected direct latency gain is small, but it
is now cheap to test and may change draft acceptance. Measure both speed and
the accepted-length histogram.

### 5. Consider adaptive depth only after the above

Depth 4 has the best median, depth 2 has a lower ceiling, and depth 8 loses on
acceptance. A depth controller could demote after short partial accepts and
promote after repeated full accepts, but the scheduler's 16-cycle window
already solves the larger prompt-level on/off problem. Do not add controller
complexity without a repeatable prompt suite.

## Useful commands

Build:

```sh
rtk make -j8 ds4-server tests/test_qwen4_metal
```

Run Metal fixtures:

```sh
rtk ./tests/test_qwen4_metal
```

Run the current best model with exact depth-4 MTP and timing:

```sh
rtk env \
  DS4_QWEN4_EXPERIMENTAL_Q4_DENSE=1 \
  DS4_QWEN4_EXPERIMENTAL_Q4_0_ROUTED=1 \
  DS4_QWEN4_MTP_SCHEDULER=0 \
  DS4_MTP_TIMING=1 \
  ./ds4-server \
  --model /Users/ifioravanti/models/qwen3.8-flash-next-q4dense-q40routed-exp2/Qwen3.8-Flash-Next-Q4KExperts-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out.gguf \
  --ple /Users/ifioravanti/models/qwen3.8-flash-next-q4dense-q40routed-exp2/Qwen3.8-Flash-Next-PLE-Q4_1.gguf \
  --mtp-model /Users/ifioravanti/models/qwen3.8-flash-next-q4dense-q40routed-exp2/qwen3.8-flash-next-q4-mtp.gguf \
  --mtp-draft 4 \
  --mtp-timing \
  --ctx 2048 \
  --host 127.0.0.1 \
  --port 18123
```

Use the default scheduler for a production-shaped run by removing
`DS4_QWEN4_MTP_SCHEDULER=0`. Its default observation window is now 16 cycles.

Run a completion request:

```sh
rtk curl -sS http://127.0.0.1:18123/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","prompt":"Continue the sequence: one, two, three,","max_tokens":64,"temperature":0,"ignore_eos":true}'
```

Final validation before handoff or commit:

```sh
rtk make -j8 test-qwen4-release
rtk ./tests/test_qwen4_metal
rtk git diff --check
rtk git status --short
```

## Known cleanup/documentation items

- The graph allocation log still says `MoE-down=Q4_K-expert-split` when the
  experimental base routed tensors are Q4_0. Make the log report the bound
  routed qtype. (DONE, session fifteen item 20f.)
- Add the Q4_0 routed profile, its quality tradeoff, the 16-cycle scheduler
  default, and the retained tiny-verifier fusions to `QWEN38_FLASH_NEXT.md`
  after their final test gates. (DONE across sessions 15-20.)
- Keep experimental profiles guarded; do not silently make Q4_0 routed or
  Q4_K dense the default. (SUPERSEDED for Q4_0 routed by owner decision,
  session twenty-one item 28: Q4_0 routed is the STANDARD profile and the
  env is removed; Q4_K dense GDN/QSA stays experimental behind
  `DS4_QWEN4_EXPERIMENTAL_Q4_DENSE=1`.)

## Clean-context summary

The standard profile is the former `qwen3.8-flash-next-q40routed-v3dense-exp`
pack (v3
dense Q8 GDN/QSA + Q4_0 routed; the directory name is historical).  The
`DS4_QWEN4_EXPERIMENTAL_Q4_0_ROUTED` env is REMOVED — the loader accepts
Q4_0-routed packs unconditionally and detects the variant from the actual
tensor types (item 28).  Session sixteen fixed the cold-PLE first pass
(parallel pread
gather, `DS4_QWEN4_PLE_GATHER_THREADS`, cold 637 -> 1054 tok/s paired)
and tiled the BF16 control matmuls (router + GDN decay/beta, 767-801 ->
66 ms per 8K chunk; `DS4_QWEN4_BF16_MUL_MM_MIN_ROWS=0` restores
scalar), lifting warm prefill to the 940-1090 band.  Session seventeen
landed two more bit-exact default-on kernels for the QSA streaming
top-k scan: the register-batched scorer
(`DS4_QWEN4_QSA_SCORE_BATCHED=0` restores the old grid; 734 -> 138
ms/layer at 64K visible blocks) and the threshold-filtered merge-select
(`DS4_QWEN4_QSA_MERGE_SELECT=0` restores the full-sort merge; 253 ->
~60 ms/layer) — the stage is 5.4x faster, short-frontier sweeps gain
5.9-6.9 percent, and the whole-258048-token frontier went 625.9 ->
937-938 tok/s (+50 percent) with decode unchanged.  Session eighteen
landed the tensor-core index scorer `kernel_qwen4_qsa_score_tile_mm_
bf16` (DEFAULT ON; `DS4_QWEN4_QSA_SCORE_MM=0` restores the scalar
batched scorer; `DS4_QWEN4_QSA_SCORE_MM_MIN_QUERIES`, default 16,
keeps decode/verifier rows scalar): the four index heads fold into one
F16-operand GEMM on the matrix units — drift-gated, not byte-exact.
Measured: scorer 134-154 -> 46-48 ms/layer at 64K visible blocks
(2.8-3.2x, ~11.5 TFLOP/s effective), fresh-chunk scan 16.1 -> 13.6
ms/layer; full model 4K-interval sweeps +0.2 to +1.6 percent (growing
with frontier) and the whole-258048 frontier 963.8 -> 1017.7 tok/s
pair means (+5.6 percent, both interleaved pairs positive) with decode
unchanged; drift vs the exact-checkpoint fixture +0.0000082 MAE
(0.044404 against the 0.044396 anchor, 250x inside budget), 16K
logits delta 0.083/0.66 vs the accepted 0.074/0.55 chunk-size
envelope.  The 16K/128 greedy anchor is now
35ac916d9472f2db303f069ee31bdcca69021227e39a24a8b903b73e9dd2e8ea at
defaults (the pre-session f6c2a929... remains reachable with
DS4_QWEN4_QSA_SCORE_MM=0 and reproduces exactly).  All five changes
are fixture-pinned and kill-switchable; make test-qwen4 aggregate is
green (126 PASS).  Remaining prefill levers are small: the
merge-select is near its floor, the MMA scorer sits ~2x under the
dense mul_mm ceiling (short K=128 amortization), and gathered
attention, the dense Q8_0 family, MoE, GDN, and BF16 controls are at
their measured ceilings (items 21-24).  DS4_METAL_GPU_STAGE_PROFILE=1
plus DS4_QWEN4_PROFILE=1 reproduces the per-label GPU table on any
run (prefill only — the per-dispatch flush perturbs decode; and
DS4_QWEN4_QSA_TOPK_ABLATE=score|merge splits the streaming top-k
label).

Session nineteen (item 26) then re-ran the opt-in GQA MMA attention
A/B on the leaner chunk: the speed win is REAL (fixed kernel 1.44x
standalone, sweep +4.3 percent, 258048 pairs +4.3-9.0 percent) — but
the round exposed that the kernel had SHIPPED COMPUTING UNIFORM
SOFTMAX since session fourteen (an unguarded sfrag store let the PV
simdgroups' uninitialized fragments clobber the score groups' s_tile
rows; every small-magnitude fixture passed because uniform
approximates true weights there).  The bug is fixed and the fixture
hardened with production-magnitude cases (negative-tested), and an
opt-in F16 hi/lo Q-split variant was added and measured: the
corrected kernel's residual drift (+0.0137 fixture MAE over the
0.0444 anchor; 16K logits 1.21/11.06) is P-tile F16 rounding plus
MMA accumulation order, NOT Q precision — the Q-split leaves it
unchanged (0.0582 vs 0.0581) while costing the entire speed win
standalone.  Gathered attention is therefore CLOSED on this model:
F16-operand attention cannot hold the quality class at a speed that
beats the scalar kernel; the scalar path remains the default and the
MMA path stays opt-in for study (`DS4_QWEN4_QSA_GQA_MMA=1`, plus
`DS4_QWEN4_QSA_GQA_MMA_QSPLIT=1` for the split variant).  Default
state is byte-identical throughout (both anchors reproduce on the
final binary) and the aggregate stays green.

Session twenty (item 27) then ran the first MTP depth/margin sweep on
the v3dense quality profile, config-only (the tree is byte-identical
to the session-nineteen handoff): replay-free MTP is a
WORKLOAD-DEPENDENT win that the shipped scheduler arbitrates
correctly.  Plain decode re-baselined at 35.3-35.5 tok/s short / 33.1
at ~8K context; the verify pass on this pack is 37-41 percent cheaper
than the v3 pack's (48.4 vs 77.6-82.5 ms at depth 4 — the Q4_0-routed
win carries into verification), and the depth optimum is
family-dependent with a measured boundary (owner-directed 5-8
extension): 7 on the deterministic continuation (56.3 mean / 57.8 peak
tok/s, +57 percent over plain — the machine record for this model),
5-6 on factual lists (48.3-48.5, +26 to +37 percent), and depth 8 is
worse on both engaged families — below d7 on chat and scheduler-
BYPASSED on factual — so the v3-pack depth-8 rejection transfers,
albeit for the opposite reason (marginal verify-row EV, not acceptance
collapse).  On novel-prose continuation the draft's
per-position agreement is only ~50-60 percent at ANY context length,
forced MTP loses 8-21 percent on both v3dense AND the original v3 pack
(same-slice controls), and the default scheduler bypasses every such
request after its 16-cycle window at actual/baseline 1.15-1.9 — the
bounded 0.4-3.8 percent window cost (non-monotone in depth: the 1.5x
severe-loss check trips earlier for deeper drafts).  The record's
historic acceptance numbers (3.65 committed, 2.65 accepted,
engaged-at-8.7K) are prompt-construction properties, not pack
properties.  Margin 110 is
validated from both sides (engagement ratio ~0.75 where profitable;
bypass ratios 1.15-1.9 where not; any margin in [100,120] decides
identically, and higher margins would force measured losses) — no
change.  The secondary decode re-attribution (CPU buckets + relative
stage profile) confirms the items 8-12 picture on the current binary:
the M=1 window is seven comparable latency-bound stages (QSA 21,
MoE 19, GDN 18, MLP-HC-read 15, attention-HC-read 14, MLP-HC-write 11,
attention-HC-write 1 percent) and verifier rows scale sub-linearly —
no new kernel direction.  Recommended configuration, documented in
QWEN38_FLASH_NEXT.md: enable the sidecar with `--mtp-draft 5` (balanced;
`7` for known-predictable workloads, never 8) and the
default scheduler for mixed/predictable workloads; expect +26 to +57
percent there and a bounded sub-4-percent cost on pure prose.  Driver
archived at speed-bench/qwen4_mtp_sweep.py.
