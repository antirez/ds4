# Session handoff — DS4 GB10 CUDA decode optimization

Last updated: 2026-05-12 (post-F16-vec8 session). Read this first.

## Where we are right now

| Metric | Value |
| --- | --- |
| Active model | `fixed-imatrix-b0c3326/...-chat-v2-imatrix.gguf` (via `/home/cghart/ds4/ds4flash.gguf` symlink — still needs manual swap, see below) |
| **Baseline gen tok/s @ ctx 7047** | **15.00** (no MTP, all session optimizations on) |
| Session-start baseline (all this-session flags disabled) | 13.66 |
| Session delta | **+1.34 tps, -6.09 ms/token total** |
| `--logprob-vectors` against `local.vec` | **OK** |
| `--bench-mtp-spec-source --bench-exact-replay-source --cuda-indexed-decode-heads8-source --decode-profile-source` | OK |
| Source state | three uncommitted changes (out_a/b fuse + F16 pair vec8 + F16 single vec8), all gated on disable env flags |

## What changed this session (2026-05-12 — F16-vec8)

Two evidence-backed kernel-level wins. Full breakdown in
`docs/hardware/gb10-cuda-notes.md` under
**"F16 GEMV vectorization wins (2026-05-12, post-q_a/kv-pair-fusion)"**.

### Lever 1: output_a/output_b/HC-expand sequential fuse — small, real

`grouped_q8_0_a_preq_to_q8_kernel` + `ds4_gpu_attention_output_q8_fused_hc_tensor`
write `low` directly as Q8_0 layout, skipping one
`quantize_q8_0_f32_kernel` launch and the float roundtrip between
`attn_output_a` and `attn_output_b`. **First design (8 warps × 4 outputs/warp)
regressed** because it cut CTA count to 1/4 and added intra-warp sequential
work; final design (1024 threads/CTA, 32 warps × 1 output each) preserves
the original per-warp parallelism. Win: +0.07 tps, -0.33 ms output_proj.

Disable: `DS4_CUDA_DISABLE_OUT_AB_FUSE=1`.

### Lever 2: F16 GEMV uint4 vectorization — the big lever

`matmul_f16_pair_ordered_chunks_kernel` and
`matmul_f16_ordered_chunks_kernel` were doing scalar `__half2float(wr[i])`
reads with a shared-memory + serial-32 reduction; both were at ~44% of
GB10's sequential-read ceiling. Added
`matmul_f16_pair_warp_vec8_kernel` and `matmul_f16_warp_vec8_kernel`:
`uint4` (16-byte) weight loads = 8 halves per LSU op, `__half22float2`
unpack, `warp_sum_f32` (warp-shuffle) reduction.

Bench delta: compressor 5.51 → 2.94 ms (~211 GB/s, 86% of peak); also
helps `attn_hc_pre` and `shared_ffn` because the `ordered_router` branch
covers every single-token F16 GEMV in the decode loop. Total:
+1.27 tps, -5.48 ms total_ms/token.

Disable: `DS4_CUDA_NO_F16_PAIR_VEC8=1`, `DS4_CUDA_NO_F16_VEC8=1`.

### Falsification of prior session's calibration note

Prior handoff established a "3-5× optimism bias on kernel-side estimates"
warning based on MoE and q8_0 fusion data. That was real for those
kernels, but it was conditioned on specific evidence sets, not a universal
multiplier. F16 GEMV vectorization went the *other* way: estimate "F16
already well-tuned, no notebook entry" — actual +1.3 tps. Each new probe
should set its own theoretical floor before measuring; don't apply a
session-old optimism multiplier to a different kernel family.

### Step 3 (CUDA Graphs) — deferred with rationale

Scoping showed GPU is compute-bound (~67 ms/token GPU vs ~5 ms CPU enqueue,
fully overlapped on the spin-scheduled path). Estimated CUDA-Graph gain
0.1-0.3 tps for a real structural refactor. **Deferred.** Concrete entry
point for the next session is in `gb10-cuda-notes.md` "Step 3" — start
with a GPU-idle-window probe before committing to the graph capture.

## MTP background (unchanged from prior session)

**Prior plan claimed Q4_K vectorization would unlock 16–19 tok/s.**
Q4_K kernel was vectorized with `dp4a` (see `ds4_cuda.cu::dev_dot_q4_32`
and `dev_dot_q4_K_q8_K_block`). Correctness preserved, α=0.571 unchanged.
**The plan's math was wrong.** Profiling reveals MTP can't reach 20 tps
on this architecture — see below.

## MTP isn't the lever. The real bottleneck is the verifier structure.

Measured MTP cycle (steady-state, 64 tokens, draft=2, strict, with vectorized Q4_K):

| Cycle step | Cost | Notes |
| --- | --- | --- |
| Vanilla decode of accepted token (in `ds4_session_eval`) | ~74 ms | unavoidable |
| MTP draft inside `ds4_session_eval` line 17463 | ~21 ms | sets up next-cycle `drafts[0]` |
| MTP draft in spec for-loop line 17616 | ~21 ms | produces `drafts[1]` |
| `decode2_exact` verify (2 positions) | ~165 ms | **runs both positions SERIALLY per layer (line 13733-13770), no weight sharing** |
| Snapshot / restore / misc | ~5 ms | KV-state copies |
| **Per-cycle total** | **~286 ms** for max 2 tokens = ~7 tps theoretical best |
| **Measured** | **3.11 tps** (extra time from reject-fallback decodes) |

`DS4_MTP_BATCH_VERIFY=1` (uses `metal_graph_verify_suffix_tops` via the
prefill batch-layer path) gave only 143 ms vs 165 ms — 13% improvement,
not the ~50% one would expect from true weight sharing. So even the
nominally-batched path isn't really weight-sharing across the 2 rows in
the dimensions where it could.

**Theoretical ceiling, even with a true weight-sharing batch verifier and α=1.0: ~12 tps.**
MoE (23% of decode cost) doesn't batch-share — different experts per token.
That hard limit makes MTP at draft=2 a poor fit for this model on this
hardware regardless of further drafter/Q4_K work.

## Highest-EV next action

**Decode breakdown after this session's wins**
(`DS4_CUDA_ATTENTION_EVENT_PROFILE=1 DS4_CUDA_DECODE_EVENT_PROFILE=1`,
ctx 7047, 128 gen-tokens, fixed-imatrix model):

| Stage | ms/token (post-vec8) | Δ from prev | % of decode |
| --- | --- | --- | --- |
| Attention (q_b + out_a/b + main kernel + hc_pre) | **33.2** | -1.5 | 50% |
| MoE | 16.1 | -0.6 | 24% |
| Compressor + indexer | **6.5** | **-4.5** | 10% |
| Shared FFN | 8.8 | -0.8 | 13% |
| Output logits | 2.6 | 0.0 | 4% |
| **Total** | **67.07** | **-6.09** | |

Compressor stage shrunk dramatically; q_proj and the attention kernel are
now the dominant non-MoE substages.

**Next session candidates, ordered by EV:**

1. **q8_0 GEMV vectorization audit (1 session, est +0.0-0.3 tps)** —
   Check whether `matmul_q8_0_preq_warp8_kernel` and the `_hc_expand_`
   family have the same scalar-load underutilization that F16 had. Most
   of the q8_0 path was hand-tuned for dp4a already, but some scalar
   loads survive. Target shapes: `attn_q_b` (q_rank=1024 →
   N_HEAD×HEAD_DIM=32768), `attn_output_b` (8192 → 4096),
   `hc_ffn_fn` (4096 → mix_hc).
2. **Compressor_update kernel review (1 session, est +0.0-0.5 tps)** —
   The compressor stage still has ~4.2 ms of uninstrumented work
   (APE + norm + EMA + FP8 quantize). Split it into substages first,
   then look for a kernel-level lever. Bias to "no win here" — these
   are compute-heavy not bandwidth-bound — but quick to confirm.
3. **GPU-idle-window probe → CUDA Graphs (1-2 sessions, est +0.1-0.3 tps)** —
   See `gb10-cuda-notes.md` "Step 3" for the concrete first probe. Only
   commit to graphs if the host-side window measurement justifies it.

## Open work, ordered by EV

| Item | Effort | Expected tok/s | Notes |
| --- | --- | --- | --- |
| **q8_0 GEMV vectorization audit** | 1 session | +0.0–0.3 tps | Pattern-match the F16 win onto the remaining q8_0 GEMVs (attn_q_b, etc.) |
| Compressor_update substage split + review | 1 session | +0.0–0.5 tps | Uninstrumented ~4 ms in compressor stage |
| GPU-idle-window probe + (maybe) CUDA Graphs | 1-2 sessions | +0.1–0.3 tps | Verify gap before refactoring |
| Drop redundant MTP draft inside `ds4_session_eval` | 4 hrs | +0.1 tps (only when MTP on) | Modest correctness risk; defer |
| Build a true weight-sharing batch verifier | weeks | unlocks MTP as a path | Strategic; not session-scale |
| Multi-fd cache (thorough fd-cache fix) | 2-4 hrs | ~0 in steady state | Code hygiene, ~3.8 GB host page-cache savings |
| Land MTP head bit-equivalence test | 2-3 hrs | None | Regression gate for future MTP work |

## Files that landed this session

| File | Purpose |
| --- | --- |
| `docs/SESSION_HANDOFF.md` | This file. Start here. |
| `docs/mtp-fdcache-bug-report.md` | Shareable upstream bug report (root cause + fix + repro) |
| `docs/hardware/gb10-cuda-notes.md` | Durable notebook; all measurements + decisions across sessions |
| `tests/test-vectors/regen_local_vectors.py` | Regenerate `local.vec` for a new model (cloud-fixture-independent regression gate) |
| `tests/test-vectors/local.vec` | Local-model logprob fixture (replaces cloud `official.vec` when the imatrix changes) |
| `tests/test-vectors/README.md` | Updated with the `local.vec` workflow |

## In-source changes (durable, not stripped)

### This session (2026-05-12, F16-vec8) — uncommitted

| File | Change | Purpose |
| --- | --- | --- |
| `ds4_cuda.cu` | `grouped_q8_0_a_preq_to_q8_kernel` (1024-thread/CTA) | Writes Q8_0 layout directly from grouped output_a; skips one prequant launch + the float `low` DRAM roundtrip |
| `ds4_cuda.cu` | `ds4_gpu_attention_output_q8_fused_hc_tensor` | Host launcher that calls the fused kernel then the existing `hc_expand_preq_warp8_unroll<256>` directly |
| `ds4_cuda.cu` | `matmul_f16_pair_warp_vec8_kernel` + `matmul_f16_warp_vec8_kernel` | `uint4` 8-half loads + `__half22float2` + warp_sum_f32; replaces the scalar-load chunks kernels on `in_dim % 8 == 0` |
| `ds4_cuda.cu` | Dispatch in `ds4_gpu_matmul_f16_pair_tensor_impl` + `ds4_gpu_matmul_f16_tensor` (ordered_router branch) | Default-on selection of the vec8 variants |
| `ds4_gpu.h` | `ds4_gpu_attention_output_q8_fused_hc_tensor` prototype | New extern entry |
| `ds4.c` | `metal_graph_disable_out_ab_fuse()` + call-site switch in `metal_graph_encode_decode_layer` | Wires the fused output path into the production decode flow |

### Prior sessions (durable, already shipped)

| File | Change | Purpose |
| --- | --- | --- |
| `ds4_cuda.cu` | `cudaSetDeviceFlags(cudaDeviceScheduleSpin)` in `ds4_gpu_init` | Match llama.cpp Blackwell decode tuning; flag stays even though it's neutral on GB10 today |
| `ds4_cuda.cu` | `moe_gate_up_mid_decode_lut_qwarp32_unroll_kernel<XQ_BLOCKS>` + dispatch | +0.7% from compile-time unroll of the iq2 inner loop; default on for `xq_blocks=16` |
| `ds4_cuda.cu` | `float2` reads in `attention_indexed_mixed_kernel` + `attention_decode_mixed_kernel` phase-3 fast path | Tiny win on attention; always on |
| `ds4_cuda.cu` | `cuda_block_q4_K` + `dev_q4_K_get_scale_min` + `dev_dot_q4_K_q8_K_block` + `dev_dot_q4_32` | Q4_K decode primitives; dp4a-vectorized (8x `__dp4a` per sub-block, unrolled outer loop) |
| `ds4_cuda.cu` | `moe_gate_up_mid_decode_q4K_qwarp32_kernel` + `moe_down_q4K_sum6_qwarp32_kernel` | Q4_K MoE kernel variants for MTP draft head |
| `ds4_cuda.cu` | `routed_moe_launch` accepts `(gate=12, down=12)` Q4_K | Unblocks MTP routed MoE |
| `ds4_cuda.cu` | **fd-cache host-base check** in `cuda_model_range_ptr_from_fd` + `g_model_fd_host_base` | **The MTP α=0 root cause fix.** See `mtp-fdcache-bug-report.md`. |
| `ds4_cuda.cu` | `ds4_gpu_memfloor_bench` + `ds4_check_nan_kernel` + `ds4_gpu_check_nan` | Bench infra. Memfloor is permanent; NaN check is env-gated diagnostic. |
| `ds4.c` | NaN probe calls in MTP draft chain (env-gated by `DS4_MTP_NAN_PROBE`) | Diagnostic; zero cost when off |
| `ds4.c` | `DS4_MTP_VERIFY_LOG`, `DS4_MTP_CHAIN_LOG` instrumentation | Diagnostic; zero cost when off |
| `tests/ds4_test.c` | Source-hook assertions for new env flags + memfloor bench test | CI gate |

## Reproduction commands

```sh
# Build
make ds4-bench ds4_test ds4

# Baseline (no MTP)
./ds4-bench --cuda -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --gen-tokens 128

# Correctness gate against local fixture
DS4_TEST_MODEL=/home/cghart/ds4/ds4flash.gguf \
DS4_TEST_VECTOR_FILE=tests/test-vectors/local.vec \
  ./ds4_test --logprob-vectors

# MTP α measurement (the breakthrough run)
DS4_MTP_STRICT=1 DS4_MTP_RESTORE_DRAFT_FRONTIER=1 \
  ./ds4-bench --cuda -m /home/cghart/ds4/ds4flash.gguf \
  --mtp /home/cghart/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --mtp-draft 2 \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --gen-tokens 16
# Expect: draft_accept_rate=0.571, avg_accepted_per_cycle=2.000

# All source-hook tests
./ds4_test --bench-mtp-spec-source --bench-exact-replay-source \
  --cuda-indexed-decode-heads8-source --decode-profile-source

# Regenerate local fixture (if the model changes)
./tests/test-vectors/regen_local_vectors.py \
  -m /home/cghart/ds4/ds4flash.gguf \
  -o tests/test-vectors/local.vec
```

## Symlink note

The session ran against the new fixed-imatrix model via explicit `-m` paths
because the Claude harness blocked the symlink swap (it sits outside the
project tree). To make every command in this handoff "just work", run once
in your shell:

```sh
ln -sfn gguf/fixed-imatrix-b0c3326/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  /home/cghart/ds4/ds4flash.gguf
```

Until you do, every command above that uses `/home/cghart/ds4/ds4flash.gguf`
points at the **old** May-9 model. Easiest workaround: keep the symlink
swap and assume the new model going forward.

## Env-flag cheat sheet

### Disable-flags for this-session changes (default off → wins enabled)

| Flag | Purpose |
| --- | --- |
| `DS4_CUDA_DISABLE_OUT_AB_FUSE=1` | Disable the fused output_a/output_b/HC-expand path; restores the two-call sequence |
| `DS4_CUDA_NO_F16_PAIR_VEC8=1` | Force the scalar `matmul_f16_pair_ordered_chunks_kernel` instead of the new `_warp_vec8_kernel` |
| `DS4_CUDA_NO_F16_VEC8=1` | Same, but for the single-output F16 GEMV (ordered_router branch) |

### Diagnostics (from prior sessions, all zero-cost when off)

| Flag | Purpose |
| --- | --- |
| `DS4_MTP_VERIFY_LOG=1` | Per-cycle MTP gate/draft/verify state |
| `DS4_MTP_CHAIN_LOG=1` | Per-step status inside `metal_graph_eval_mtp_draft_from_hc`, also logs `routed_moe` quant-type rejections |
| `DS4_MTP_NAN_PROBE=1` | Host-side NaN scan of `mtp_embed`/`mtp_enorm`/`mtp_eproj`/etc; pinpoints the upstream NaN source if one returns |
| `DS4_MTP_FDCACHE_DBG=1` | Trace fd-cache bypasses (verifies the host-base fix is firing for MTP weights) |
| `DS4_CUDA_DECODE_EVENT_PROFILE=1` | CUDA event-based decode stage profile (low overhead) |
| `DS4_CUDA_MOE_EVENT_PROFILE=1` | Aggregate MoE achieved GB/s per token |
| `DS4_CUDA_INDEXER_EVENT_PROFILE=1` | Compressor + indexer sub-stage timing |
| `DS4_TEST_MEMFLOOR=1` | Run the `./ds4_test --memfloor-bench` microbench |

All zero-cost when off.

## Pointers into the hardware notebook

`docs/hardware/gb10-cuda-notes.md` is the durable log. Most relevant
sections:

- **"Decode Bandwidth Floor (2026-05-11 investigation)"** — first hardware
  bandwidth accounting
- **"Decode baseline refresh"** — current session's baseline numbers
- **"CUDA event timing for decode stages"** — Task 2 (host vs event profile)
- **"MoE bandwidth event-profile microbench (Task 4)"** — 115 GB/s achieved
  vs 245 GB/s sustained ceiling
- **"Phase 1+++"** — MoE iq2 unroll and attention float2 wins
- **"MTP α=0 root cause (2026-05-11)"** — the diagnostic trail that
  identified the fd-cache bug

## What to skip when starting fresh

- **Don't redo the kernel BW analysis.** Already in the notebook. The MoE
  iq2 and Q4_K kernels are the two with real headroom.
- **Don't try heads8 attention at B=1.** Already rejected; recorded in
  notebook with measurements.
- **Don't try the H2 vectorization on compressor pair.** Broke
  `--logprob-vectors` precision; reverted.
- **Don't re-quantize the MTP head to escape Q4_K.** The bug was not in
  Q4_K decode; it was in the fd-cache. Q4_K decode is correct.
- **Don't chase further MTP speedups via Q4_K tile/batch tweaks.** Q4_K
  is in the MTP head only (3.6 GB of 81 GB total weights). The MTP cycle
  bottleneck is the verifier doing 2 positions serially per layer, not
  Q4_K decode speed. See "MTP isn't the lever" above.

## Confidence calibration from this session (F16-vec8)

Predicted vs actual:
- "Output_a/b fusion ceiling 0.3-0.5 ms" — **right** (0.33 ms in
  output_proj substage). Calibration-note pessimism (3-5× optimism bias)
  did not apply here; the notebook's own number was on the nose.
- "F16 GEMV was already tuned" (implicit, no entry in notes) — **wrong**.
  Scalar `__half2float` loads left both pair and single F16 kernels at
  44% of peak bandwidth. uint4 vectorization recovered most of the gap.
- "CUDA Graphs gives 1-3 ms" — **untested but downgraded after scoping**.
  Per-launch overhead is ~5 µs × ~1000 launches = ~5 ms CPU enqueue,
  but the GPU at 67 ms/token absorbs that easily. Realistic gain is
  whatever GPU-idle gaps exist between back-to-back short kernels; not
  yet measured.
- "Calibration note's 3-5× optimism bias is a universal warning" —
  **partially wrong**. It held for the kernels it was derived from
  (MoE, q8_0 fusion) but didn't apply to F16 GEMV. Per-probe theoretical
  floors beat carried-forward heuristics.

## Confidence calibration from prior session (kept for reference)

- "Q4_K decode bug" — wrong (turned out the math was correct; the upstream cache bug was the culprit)
- "MTP definitively closed" — wrong (it works at α=0.571 with the fd-cache fix)
- "+0.2-0.4 tok/s per kernel-only session" — was right then; this session blew through it (+1.34 tps)
- "EAGLE-class drafter required for speculation" — wrong premise; the MTP head drafter is healthy.
- **"Q4_K vectorization unlocks 16-19 tok/s"** — **wrong** (post-Q4_K-vectorize profiling shows the MTP cycle is verifier-bound, not draft-bound; effective_tps≈3.1, no improvement from Q4_K speed).

Net lesson: do the bandwidth+structure accounting per cycle before
committing to a multi-day plan. Don't let one session's calibration note
override per-probe theoretical floors in unrelated kernel families.
